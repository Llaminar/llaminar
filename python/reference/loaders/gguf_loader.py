"""
GGUF Loader

Main orchestrator for loading GGUF files into PyTorch state dicts.

Orchestrates the complete pipeline:
1. Parse GGUF file (header, metadata, tensor info)
2. Dequantize tensor data (Q4_0, Q8_0, Q6_K, F16 → FP32)
3. Map tensor names (GGUF → HuggingFace conventions)
4. Build PyTorch state_dict

Usage:
    loader = GGUFLoader("model.gguf")
    config, state_dict = loader.load()
    
    # With transformers
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)
    
    # Without transformers (just get config dict)
    config_dict, state_dict = loader.load(as_transformers_config=False)

Author: David Sanftenberg
"""

from pathlib import Path
from typing import Dict, Tuple, Any, Optional, Union
import sys

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not available. GGUFLoader will return NumPy arrays instead.", 
          file=sys.stderr)

try:
    from transformers import PretrainedConfig, Qwen2Config, LlamaConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    PretrainedConfig = None
    Qwen2Config = None
    LlamaConfig = None

from .gguf_parser import GGUFParser, GGUFTensorInfo
from . import dequantize
from .tensor_name_mapper import TensorNameMapper, create_mapper_from_metadata


class GGUFLoader:
    """
    Main GGUF loader class.
    
    Loads GGUF files into PyTorch-compatible state dicts with optional
    HuggingFace transformers config extraction.
    
    Args:
        file_path: Path to GGUF file
        verbose: Print progress information
        
    Example:
        loader = GGUFLoader("qwen2.5-0.5b-instruct-q4_0.gguf")
        config, state_dict = loader.load()
    """
    
    def __init__(self, file_path: Union[str, Path], verbose: bool = False):
        self.file_path = Path(file_path)
        self.verbose = verbose
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.file_path}")
    
    def _log(self, message: str):
        """Print message if verbose mode enabled"""
        if self.verbose:
            print(message)
    
    def load_config(
        self, 
        parser: Optional[GGUFParser] = None,
        as_transformers_config: bool = True
    ) -> Union[Any, Dict[str, Any]]:
        """
        Extract model configuration from GGUF metadata.
        
        Args:
            parser: Optional pre-initialized parser (otherwise creates one)
            as_transformers_config: Return transformers PretrainedConfig if available
            
        Returns:
            PretrainedConfig (if transformers available) or dict
            
        Raises:
            RuntimeError: If config extraction fails
        """
        # Create parser if not provided
        close_parser = parser is None
        if parser is None:
            parser = GGUFParser(str(self.file_path))
            parser.parse()
        
        try:
            # Get base config dict from parser
            config_dict = parser.get_config_dict()
            model_type = parser.get_model_type()
            
            self._log(f"Detected model type: {model_type}")
            self._log(f"Config keys: {list(config_dict.keys())}")
            
            # Add model_type to config
            if model_type:
                config_dict['model_type'] = model_type
            
            # Add architecture-specific keys from metadata
            if model_type:
                prefix = f"{model_type}."
                
                # Additional keys that might be needed
                additional_keys = {
                    'attention.head_count_kv': 'num_key_value_heads',
                    'rope.dimension_count': 'rope_dim',
                    'rope.freq_base': 'rope_theta',
                    'attention.layer_norm_rms_epsilon': 'rms_norm_eps',
                }
                
                for gguf_key, hf_key in additional_keys.items():
                    full_key = prefix + gguf_key
                    if full_key in parser.metadata:
                        config_dict[hf_key] = parser.metadata[full_key]
            
            # Return as transformers config if requested and available
            if as_transformers_config and HAS_TRANSFORMERS:
                if model_type == 'qwen2':
                    return Qwen2Config(**config_dict)
                elif model_type in ('llama', 'llama2', 'llama3'):
                    return LlamaConfig(**config_dict)
                else:
                    # Generic PretrainedConfig
                    config_dict.setdefault('model_type', 'unknown')
                    from transformers import PretrainedConfig
                    return PretrainedConfig(**config_dict)
            else:
                return config_dict
                
        finally:
            if close_parser:
                parser.close()
    
    def load_state_dict(
        self, 
        parser: Optional[GGUFParser] = None,
        as_torch: bool = True,
        show_progress: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Load all tensors from GGUF file into a state dict.
        
        Args:
            parser: Optional pre-initialized parser
            as_torch: Convert to torch.Tensor (otherwise NumPy arrays)
            show_progress: Show progress bar (default: use verbose setting)
            
        Returns:
            State dict mapping HuggingFace names to tensors
            
        Raises:
            RuntimeError: If loading fails
        """
        if show_progress is None:
            show_progress = self.verbose
        
        # Create parser if not provided
        close_parser = parser is None
        if parser is None:
            parser = GGUFParser(str(self.file_path))
            parser.parse()
        
        try:
            # Create name mapper
            mapper = create_mapper_from_metadata(parser.metadata)
            
            state_dict = {}
            total_tensors = len(parser.tensors)
            
            self._log(f"Loading {total_tensors} tensors...")
            
            # Progress tracking
            unmapped_count = 0
            dequant_stats = {
                'F32': 0,
                'F16': 0,
                'Q4_0': 0,
                'Q8_0': 0,
                'Q6_K': 0,
                'other': 0
            }
            
            for i, tensor_info in enumerate(parser.tensors):
                # Progress message
                if show_progress and (i % 50 == 0 or i == total_tensors - 1):
                    print(f"  Loading tensor {i+1}/{total_tensors}: {tensor_info.name}")
                
                # Read raw tensor data
                raw_data = parser.read_tensor_data(tensor_info)
                
                # Dequantize to FP32
                fp32_array = dequantize.dequantize(
                    raw_data, 
                    tensor_info.type, 
                    tensor_info.shape
                )
                
                # Track dequantization stats
                type_name = tensor_info.type.name
                if type_name in dequant_stats:
                    dequant_stats[type_name] += 1
                else:
                    dequant_stats['other'] += 1
                
                # Map GGUF name to HuggingFace name
                hf_name = mapper.map_name(tensor_info.name)
                
                # Track unmapped tensors
                if hf_name == tensor_info.name:
                    # Name unchanged - possibly unmapped
                    if not any(hf_name.startswith(p) for p in ['model.', 'lm_head']):
                        unmapped_count += 1
                        if self.verbose:
                            print(f"  WARNING: Tensor may be unmapped: {tensor_info.name}")
                
                # Convert to PyTorch tensor if requested
                if as_torch:
                    if not HAS_TORCH:
                        raise RuntimeError("PyTorch not available but as_torch=True")
                    # Make a writable copy to avoid PyTorch warning about read-only arrays
                    # (memory-mapped arrays from GGUF files are read-only)
                    if not fp32_array.flags.writeable:
                        fp32_array = fp32_array.copy()
                    tensor = torch.from_numpy(fp32_array)
                else:
                    tensor = fp32_array
                
                # NOTE: As of the dimension reversal fix in gguf_parser.py, dimensions are now
                # in standard row-major order (matching PyTorch/NumPy conventions).
                # No additional transposition is needed.
                
                state_dict[hf_name] = tensor
            
            # Summary
            self._log(f"\nState dict loaded successfully:")
            self._log(f"  Total tensors: {total_tensors}")
            self._log(f"  Unmapped tensors: {unmapped_count}")
            self._log(f"  Dequantization breakdown:")
            for dtype, count in sorted(dequant_stats.items()):
                if count > 0:
                    self._log(f"    {dtype}: {count} tensors")
            
            return state_dict
            
        finally:
            if close_parser:
                parser.close()
    
    def load(
        self,
        as_transformers_config: bool = True,
        as_torch: bool = True,
        show_progress: Optional[bool] = None
    ) -> Tuple[Union[Any, Dict], Dict[str, Any]]:
        """
        Load complete GGUF file: config + state dict.
        
        This is the main entry point for loading GGUF files.
        
        Args:
            as_transformers_config: Return transformers config (vs dict)
            as_torch: Return torch tensors (vs NumPy arrays)
            show_progress: Show loading progress
            
        Returns:
            Tuple of (config, state_dict)
            - config: PretrainedConfig or dict
            - state_dict: Dict mapping tensor names to torch.Tensor or np.ndarray
            
        Example:
            loader = GGUFLoader("model.gguf", verbose=True)
            config, state_dict = loader.load()
            
            # Load into transformers model
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_config(config)
            model.load_state_dict(state_dict)
        """
        self._log(f"Loading GGUF file: {self.file_path}")
        self._log(f"  File size: {self.file_path.stat().st_size / 1024**2:.1f} MB")
        
        # Parse file once and reuse parser
        with GGUFParser(str(self.file_path)) as parser:
            parser.parse()
            
            self._log(f"\nGGUF Header:")
            self._log(f"  Version: {parser.version}")
            self._log(f"  Tensors: {parser.tensor_count}")
            self._log(f"  Metadata entries: {parser.metadata_kv_count}")
            
            # Extract config
            self._log(f"\nExtracting model configuration...")
            config = self.load_config(
                parser=parser,
                as_transformers_config=as_transformers_config
            )
            
            # Load state dict
            self._log(f"\nLoading tensors...")
            state_dict = self.load_state_dict(
                parser=parser,
                as_torch=as_torch,
                show_progress=show_progress
            )
            
            self._log(f"\n{'='*60}")
            self._log(f"GGUF LOADING COMPLETE ✓")
            self._log(f"{'='*60}")
            self._log(f"Config type: {type(config).__name__}")
            self._log(f"State dict size: {len(state_dict)} tensors")
            
            if as_torch and HAS_TORCH:
                total_params = sum(t.numel() for t in state_dict.values())
                total_size_mb = sum(t.numel() * t.element_size() for t in state_dict.values()) / 1024**2
                self._log(f"Total parameters: {total_params:,}")
                self._log(f"Total size: {total_size_mb:.1f} MB")
            
            return config, state_dict
        """Load GGUF file and return (config, state_dict)"""
        raise NotImplementedError("GGUF loader implementation pending")
