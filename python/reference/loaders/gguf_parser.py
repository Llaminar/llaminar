"""
GGUF File Format Parser

Low-level parser for GGUF (GPT-Generated Unified Format) files.
Extracts metadata, tensor information, and provides access to tensor data.

File Format Reference:
https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md

GGUF Structure:
- Header (magic, version, counts)
- Metadata KV pairs (config, tokenizer, etc.)
- Tensor info array (name, shape, type, offset)
- Alignment padding
- Tensor data section

Author: David Sanftenberg
"""

import struct
import mmap
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import IntEnum


class GGUFValueType(IntEnum):
    """GGUF metadata value types"""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGUFTensorType(IntEnum):
    """GGUF tensor data types"""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    # Q4_2 = 4  # removed upstream
    # Q4_3 = 5  # removed upstream
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    IQ1_M = 29


class GGUFTensorInfo:
    """Information about a tensor in the GGUF file"""
    
    def __init__(self, name: str, dimensions: List[int], tensor_type: GGUFTensorType, offset: int):
        self.name = name
        self.dimensions = dimensions
        self.type = tensor_type
        self.offset = offset
        
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return tensor shape as tuple"""
        return tuple(self.dimensions)
    
    @property
    def n_elements(self) -> int:
        """Calculate total number of elements"""
        n = 1
        for dim in self.dimensions:
            n *= dim
        return n
    
    @property
    def is_quantized(self) -> bool:
        """Check if tensor uses quantized format"""
        return self.type not in (GGUFTensorType.F32, GGUFTensorType.F16)
    
    def __repr__(self) -> str:
        return f"GGUFTensorInfo(name={self.name}, shape={self.shape}, type={self.type.name})"


class GGUFParser:
    """
    Parser for GGUF (GPT-Generated Unified Format) files.
    
    Provides low-level access to GGUF file structure:
    - Header information (version, counts)
    - Metadata key-value pairs (model config)
    - Tensor information (names, shapes, types, offsets)
    - Tensor data (memory-mapped for efficiency)
    
    Example:
        parser = GGUFParser("model.gguf")
        parser.parse()
        
        # Access metadata
        n_layers = parser.metadata.get("qwen2.block_count")
        
        # Access tensors
        for tensor_info in parser.tensors:
            data = parser.read_tensor_data(tensor_info)
    """
    
    GGUF_MAGIC = b"GGUF"
    ALIGNMENT = 32  # Default alignment for tensor data
    
    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize GGUF parser.
        
        Args:
            file_path: Path to GGUF file
        """
        self.file_path = Path(file_path)
        self.file = None
        self.mmap = None
        
        # Parsed data
        self.version: Optional[int] = None
        self.tensor_count: Optional[int] = None
        self.metadata_kv_count: Optional[int] = None
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[GGUFTensorInfo] = []
        self.data_offset: Optional[int] = None
        
        # Current file position during parsing
        self._offset = 0
        
    def __enter__(self):
        """Context manager entry"""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        
    def open(self):
        """Open the GGUF file for reading"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.file_path}")
        
        self.file = open(self.file_path, 'rb')
        # Memory-map for efficient large file access
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self._offset = 0
        
    def close(self):
        """Close the GGUF file"""
        if self.mmap is not None:
            self.mmap.close()
            self.mmap = None
        if self.file is not None:
            self.file.close()
            self.file = None
            
    def parse(self):
        """
        Parse the entire GGUF file structure.
        
        Parses:
        1. Header (magic, version, counts)
        2. Metadata KV pairs
        3. Tensor information
        4. Calculates data section offset
        
        Raises:
            ValueError: If file format is invalid
        """
        if self.mmap is None:
            self.open()
            
        self._parse_header()
        self._parse_metadata()
        self._parse_tensor_info()
        self._calculate_data_offset()
        
    def _read(self, fmt: str) -> Tuple:
        """Read struct-formatted data from current offset"""
        size = struct.calcsize(fmt)
        data = struct.unpack(fmt, self.mmap[self._offset:self._offset + size])
        self._offset += size
        return data
    
    def _read_string(self) -> str:
        """Read a GGUF string (uint64 length + UTF-8 data)"""
        length = self._read('<Q')[0]  # uint64 little-endian
        if length == 0:
            return ""
        string_bytes = self.mmap[self._offset:self._offset + length]
        self._offset += length
        return string_bytes.decode('utf-8')
    
    def _parse_header(self):
        """Parse GGUF file header"""
        # Read magic (4 bytes)
        magic = self.mmap[self._offset:self._offset + 4]
        self._offset += 4
        
        if magic != self.GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {magic} (expected {self.GGUF_MAGIC})")
        
        # Read version (uint32)
        self.version = self._read('<I')[0]
        
        # Read tensor count (uint64)
        self.tensor_count = self._read('<Q')[0]
        
        # Read metadata KV count (uint64)
        self.metadata_kv_count = self._read('<Q')[0]
        
        print(f"GGUF version: {self.version}")
        print(f"Tensor count: {self.tensor_count}")
        print(f"Metadata KV count: {self.metadata_kv_count}")
        
    def _parse_metadata_value(self, value_type: GGUFValueType) -> Any:
        """Parse a metadata value based on its type"""
        if value_type == GGUFValueType.UINT8:
            return self._read('<B')[0]
        elif value_type == GGUFValueType.INT8:
            return self._read('<b')[0]
        elif value_type == GGUFValueType.UINT16:
            return self._read('<H')[0]
        elif value_type == GGUFValueType.INT16:
            return self._read('<h')[0]
        elif value_type == GGUFValueType.UINT32:
            return self._read('<I')[0]
        elif value_type == GGUFValueType.INT32:
            return self._read('<i')[0]
        elif value_type == GGUFValueType.FLOAT32:
            return self._read('<f')[0]
        elif value_type == GGUFValueType.BOOL:
            return bool(self._read('<B')[0])
        elif value_type == GGUFValueType.STRING:
            return self._read_string()
        elif value_type == GGUFValueType.UINT64:
            return self._read('<Q')[0]
        elif value_type == GGUFValueType.INT64:
            return self._read('<q')[0]
        elif value_type == GGUFValueType.FLOAT64:
            return self._read('<d')[0]
        elif value_type == GGUFValueType.ARRAY:
            # Read array type and length
            array_type = GGUFValueType(self._read('<I')[0])
            array_len = self._read('<Q')[0]
            # Read array elements
            return [self._parse_metadata_value(array_type) for _ in range(array_len)]
        else:
            raise ValueError(f"Unknown GGUF value type: {value_type}")
    
    def _parse_metadata(self):
        """Parse metadata key-value pairs"""
        self.metadata = {}
        
        for _ in range(self.metadata_kv_count):
            # Read key (string)
            key = self._read_string()
            
            # Read value type (uint32)
            value_type = GGUFValueType(self._read('<I')[0])
            
            # Read value
            value = self._parse_metadata_value(value_type)
            
            self.metadata[key] = value
            
        print(f"Parsed {len(self.metadata)} metadata entries")
        
    def _parse_tensor_info(self):
        """Parse tensor information array"""
        self.tensors = []
        
        for _ in range(self.tensor_count):
            # Read tensor name (string)
            name = self._read_string()
            
            # Read number of dimensions (uint32)
            n_dims = self._read('<I')[0]
            
            # Read dimensions (uint64 array)
            # GGUF stores dimensions in REVERSE order compared to standard numpy/PyTorch convention
            # E.g., GGUF stores [896, 151936] for what should be [151936, 896]
            dimensions = [self._read('<Q')[0] for _ in range(n_dims)]
            dimensions.reverse()  # Reverse to get standard row-major order
            
            # Read tensor type (uint32)
            tensor_type = GGUFTensorType(self._read('<I')[0])
            
            # Read offset (uint64)
            offset = self._read('<Q')[0]
            
            tensor_info = GGUFTensorInfo(name, dimensions, tensor_type, offset)
            self.tensors.append(tensor_info)
            
        print(f"Parsed {len(self.tensors)} tensor infos")
        
    def _calculate_data_offset(self):
        """Calculate the offset to the tensor data section"""
        # Data section starts after header + metadata + tensor info
        # Aligned to ALIGNMENT boundary
        self.data_offset = self._offset
        
        # Align to ALIGNMENT boundary
        if self.data_offset % self.ALIGNMENT != 0:
            self.data_offset += self.ALIGNMENT - (self.data_offset % self.ALIGNMENT)
            
        print(f"Tensor data starts at offset: {self.data_offset}")
        
    def read_tensor_data(self, tensor_info: GGUFTensorInfo) -> memoryview:
        """
        Read raw tensor data from file.

        Returns a zero-copy ``memoryview`` over the memory-mapped region.
        Downstream consumers feed this into ``np.frombuffer(...)`` which
        accepts any buffer-protocol object and itself does not copy.

        PERF: Previously this returned ``bytes(self.mmap[a:b])`` which forces
        the kernel to copy the entire slice (28+ GB on 27B Q8_0 models)
        out of the page cache into a Python bytes object before the
        dequant kernel even runs. The memoryview alternative defers all
        page faulting to the actual numpy reads, which then happen in
        parallel from worker threads.

        Args:
            tensor_info: Tensor information

        Returns:
            Read-only memoryview of the raw tensor data.
        """
        if self.data_offset is None:
            raise ValueError("Must call parse() before reading tensor data")

        # Calculate actual file offset
        file_offset = self.data_offset + tensor_info.offset

        # Calculate data size based on tensor type and dimensions
        data_size = self._calculate_tensor_size(tensor_info)

        # Zero-copy view into the mmap. mmap is thread-safe for reads.
        return memoryview(self.mmap)[file_offset:file_offset + data_size]
    
    def _calculate_tensor_size(self, tensor_info: GGUFTensorInfo) -> int:
        """Calculate size in bytes of tensor data"""
        n_elements = tensor_info.n_elements
        tensor_type = tensor_info.type
        
        # Type size lookup table (bytes per element or block)
        if tensor_type == GGUFTensorType.F32:
            return n_elements * 4
        elif tensor_type == GGUFTensorType.F16:
            return n_elements * 2
        elif tensor_type == GGUFTensorType.Q4_0:
            # Q4_0: 32 elements per block, 18 bytes per block (2 scale + 16 data)
            block_size = 32
            n_blocks = (n_elements + block_size - 1) // block_size
            return n_blocks * 18
        elif tensor_type == GGUFTensorType.Q8_0:
            # Q8_0: 32 elements per block, 34 bytes per block (2 scale + 32 data)
            block_size = 32
            n_blocks = (n_elements + block_size - 1) // block_size
            return n_blocks * 34
        elif tensor_type == GGUFTensorType.Q6_K:
            # Q6_K: 256 elements per block, 210 bytes per block
            block_size = 256
            n_blocks = (n_elements + block_size - 1) // block_size
            return n_blocks * 210
        elif tensor_type == GGUFTensorType.Q5_K:
            # Q5_K: 256 elements per block, 176 bytes per block
            block_size = 256
            n_blocks = (n_elements + block_size - 1) // block_size
            return n_blocks * 176
        elif tensor_type == GGUFTensorType.Q4_K:
            # Q4_K: 256 elements per block, 144 bytes per block (2d+2dmin+12scales+128qs)
            block_size = 256
            n_blocks = (n_elements + block_size - 1) // block_size
            return n_blocks * 144
        elif tensor_type == GGUFTensorType.Q4_1:
            # Q4_1: 32 elements per block, 20 bytes per block (2 scale + 2 min + 16 data)
            block_size = 32
            n_blocks = (n_elements + block_size - 1) // block_size
            return n_blocks * 20
        elif tensor_type == GGUFTensorType.Q5_0:
            # Q5_0: 32 elements per block, 22 bytes per block
            block_size = 32
            n_blocks = (n_elements + block_size - 1) // block_size
            return n_blocks * 22
        elif tensor_type == GGUFTensorType.IQ3_S:
            # IQ3_S: 256 elements per row super-block, 110 bytes per block.
            # Use the fastest-varying dimension so row padding is accounted for.
            block_size = 256
            cols = tensor_info.shape[-1] if tensor_info.shape else n_elements
            rows = n_elements // cols if cols else 0
            n_blocks = rows * ((cols + block_size - 1) // block_size)
            return n_blocks * 110
        elif tensor_type == GGUFTensorType.IQ3_XXS:
            # IQ3_XXS: 256 elements per row super-block, 98 bytes per block.
            # Use the fastest-varying dimension so row padding is accounted for.
            block_size = 256
            cols = tensor_info.shape[-1] if tensor_info.shape else n_elements
            rows = n_elements // cols if cols else 0
            n_blocks = rows * ((cols + block_size - 1) // block_size)
            return n_blocks * 98
        else:
            # For other types, estimate conservatively
            # Most quantized types use 2-8 bits per value
            return n_elements * 2  # Conservative estimate
    
    def get_model_type(self) -> Optional[str]:
        """Extract model type from metadata (e.g., 'qwen2', 'llama')"""
        # Check common metadata keys
        for key in self.metadata:
            if key.startswith('general.architecture'):
                return self.metadata[key]
        
        # Fallback: infer from other keys
        if any(k.startswith('qwen2.') for k in self.metadata):
            return 'qwen2'
        elif any(k.startswith('llama.') for k in self.metadata):
            return 'llama'
        
        return None
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Extract model configuration from metadata.
        
        Returns dictionary with common config keys:
        - hidden_size
        - num_attention_heads
        - num_hidden_layers
        - intermediate_size
        - max_position_embeddings
        - vocab_size
        """
        model_type = self.get_model_type()
        prefix = f"{model_type}." if model_type else ""
        
        config = {}
        
        # Map GGUF metadata to HuggingFace config keys
        key_map = {
            'embedding_length': 'hidden_size',
            'attention.head_count': 'num_attention_heads',
            'block_count': 'num_hidden_layers',
            'feed_forward_length': 'intermediate_size',
            'context_length': 'max_position_embeddings',
        }
        
        for gguf_key, hf_key in key_map.items():
            full_key = prefix + gguf_key
            if full_key in self.metadata:
                config[hf_key] = self.metadata[full_key]

        # Some Qwen3.6 GGUFs are encoded as qwen35 and include trailing
        # next-token-prediction sidecar block(s) in block_count. Those blocks
        # remain in the tensor inventory for MTP, but the main PyTorch
        # reference graph must not instantiate them as ordinary decoder layers.
        nextn_depth_key = prefix + 'nextn_predict_layers'
        nextn_depth = int(self.metadata.get(nextn_depth_key, 0) or 0)
        raw_layers = int(config.get('num_hidden_layers', 0) or 0)
        if model_type in ('qwen35', 'qwen35moe') and nextn_depth > 0 and raw_layers >= nextn_depth:
            source_layer = raw_layers - nextn_depth
            nextn_tensor_name = f'blk.{source_layer}.nextn.eh_proj.weight'
            tensor_names = {getattr(t, 'name', t) for t in getattr(self, 'tensors', [])}
            if nextn_tensor_name in tensor_names:
                config['num_hidden_layers'] = source_layer
        
        # Vocab size (may be in tokenizer metadata)
        vocab_size_keys = [
            prefix + 'vocab_size',
            'tokenizer.ggml.tokens',  # Count array length
        ]
        for key in vocab_size_keys:
            if key in self.metadata:
                value = self.metadata[key]
                if isinstance(value, list):
                    config['vocab_size'] = len(value)
                else:
                    config['vocab_size'] = value
                break
        
        # Qwen 3.5 Gated Delta Net specific metadata
        if model_type in ('qwen35', 'qwen35moe'):
            qwen35_keys = {
                'attention.head_count_kv': 'num_key_value_heads',
                'attention.key_length': 'head_dim',
                'attention.layer_norm_rms_epsilon': 'rms_norm_eps',
                'rope.freq_base': 'rope_theta',
                'full_attention_interval': 'full_attention_interval',
                'ssm.conv_kernel': 'linear_conv_kernel_dim',
                'ssm.group_count': 'linear_num_key_heads',
                'ssm.inner_size': 'ssm_inner_size',
                'ssm.state_size': 'linear_value_head_dim',
                'ssm.time_step_rank': 'ssm_time_step_rank',
            }
            for gguf_key, hf_key in qwen35_keys.items():
                full_key = prefix + gguf_key
                if full_key in self.metadata:
                    config[hf_key] = self.metadata[full_key]

            # Rope dimension sections (array)
            dim_sections_key = prefix + 'rope.dimension_sections'
            if dim_sections_key in self.metadata:
                config['rope_dimension_sections'] = self.metadata[dim_sections_key]

        # Qwen 3.5 MoE specific metadata
        if model_type == 'qwen35moe':
            moe_keys = {
                'expert_count': 'num_experts',
                'expert_used_count': 'num_experts_per_tok',
                'expert_feed_forward_length': 'moe_intermediate_size',
                'expert_shared_feed_forward_length': 'shared_expert_intermediate_size',
            }
            for gguf_key, hf_key in moe_keys.items():
                full_key = prefix + gguf_key
                if full_key in self.metadata:
                    config[hf_key] = self.metadata[full_key]
        
        return config
    
    def __repr__(self) -> str:
        return f"GGUFParser(file={self.file_path}, version={self.version}, tensors={self.tensor_count})"
