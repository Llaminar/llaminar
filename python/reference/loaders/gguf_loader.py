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
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

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
    # Qwen3Config may not exist in older transformers versions
    try:
        from transformers import Qwen3Config
        HAS_QWEN3_CONFIG = True
    except ImportError:
        Qwen3Config = None
        HAS_QWEN3_CONFIG = False
except ImportError:
    HAS_TRANSFORMERS = False
    PretrainedConfig = None
    Qwen2Config = None
    Qwen3Config = None
    LlamaConfig = None
    HAS_QWEN3_CONFIG = False

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
                elif model_type == 'qwen3':
                    if HAS_QWEN3_CONFIG:
                        return Qwen3Config(**config_dict)
                    else:
                        # Fall back to Qwen2Config for older transformers
                        config_dict['model_type'] = 'qwen2'
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
            model_type = parser.get_model_type()
            
            state_dict = {}
            total_tensors = len(parser.tensors)
            
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
            
            # PERF: Parallelize raw-read + dequantize + name-mapping +
            # qwen35-transforms + torch.from_numpy across CPU cores. All of
            # these steps either release the GIL (numpy dequant kernels,
            # torch.from_numpy share-memory path) or are pure Python that
            # is short enough to overlap with bandwidth-bound work in other
            # threads. By moving them into the worker we keep the serial
            # consumer loop down to a single dict insert per tensor.
            #
            # We also stream the iterator (no list(...)) so the consumer
            # runs concurrently with workers and peak FP32 memory is
            # bounded by ``n_workers × largest_tensor`` instead of the
            # full materialized 110 GB on a 27B Q8_0 model.
            n_workers = min(os.cpu_count() or 4, 16)
            tensors_list = list(parser.tensors)

            def _process_tensor(tensor_info):
                raw = parser.read_tensor_data(tensor_info)
                fp32 = dequantize.dequantize(raw, tensor_info.type, tensor_info.shape)
                # mmap-backed views are read-only; torch.from_numpy needs a
                # writable buffer. ``.copy()`` releases the GIL for the
                # actual memcpy.
                if not fp32.flags.writeable:
                    fp32 = fp32.copy()
                # Explicitly drop the memoryview so the mmap can be closed
                # cleanly later (memoryviews hold exported pointers into
                # the mmap; any live view blocks ``mmap.close()``).
                if isinstance(raw, memoryview):
                    raw.release()

                hf_name = mapper.map_name(tensor_info.name)
                local_unmapped = (
                    hf_name == tensor_info.name
                    and not any(hf_name.startswith(p) for p in ('model.', 'lm_head'))
                )

                if as_torch:
                    if not HAS_TORCH:
                        raise RuntimeError("PyTorch not available but as_torch=True")
                    tensor = torch.from_numpy(fp32)
                    if model_type == 'qwen35':
                        tensor = self._apply_qwen35_transforms(
                            tensor, tensor_info.name, hf_name
                        )
                else:
                    tensor = fp32

                return tensor_info, hf_name, tensor, local_unmapped

            if show_progress:
                print(f"  Dequantizing {total_tensors} tensors with {n_workers} threads...")

            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                # Stream results so the consumer overlaps with worker pool.
                # ``ex.map`` preserves input order; results are yielded as
                # the next-in-order future completes.
                for tensor_info, hf_name, tensor, was_unmapped in ex.map(
                    _process_tensor, tensors_list
                ):
                    type_name = tensor_info.type.name
                    if type_name in dequant_stats:
                        dequant_stats[type_name] += 1
                    else:
                        dequant_stats['other'] += 1

                    if was_unmapped:
                        unmapped_count += 1
                        if self.verbose:
                            print(f"  WARNING: Tensor may be unmapped: {tensor_info.name}")

                    # NOTE: As of the dimension reversal fix in gguf_parser.py,
                    # dimensions are now in standard row-major order
                    # (matching PyTorch/NumPy conventions). No additional
                    # transposition is needed.
                    state_dict[hf_name] = tensor
            
            # Handle tied embeddings: if lm_head.weight is missing, copy from embeddings
            if 'lm_head.weight' not in state_dict:
                if 'model.embed_tokens.weight' in state_dict:
                    self._log(f"\n⚠ lm_head.weight missing - using tied embeddings (copying from model.embed_tokens.weight)")
                    state_dict['lm_head.weight'] = state_dict['model.embed_tokens.weight']
                else:
                    self._log(f"\n⚠ WARNING: lm_head.weight missing and no embed_tokens found!")
            
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
    
    # ------------------------------------------------------------------
    # Qwen 3.5 Gated Delta Net: tensor transforms and fused reconstruction
    # ------------------------------------------------------------------

    def _apply_qwen35_transforms(
        self,
        tensor: 'torch.Tensor',
        gguf_name: str,
        hf_name: str,
    ) -> 'torch.Tensor':
        """
        Apply Qwen 3.5 specific transforms to directly-mapped tensors.

        The llama.cpp converter applies several transforms when converting
        from HF → GGUF.  We reverse them here (GGUF → HF):
          - norm weights (except linear_attn.norm): GGUF stores w+1 → subtract 1
          - ssm_a → A_log: GGUF stores -exp(A_log) → apply log(-x)
          - ssm_conv1d: GGUF squeezes dim-1 → unsqueeze back
        """
        # Norm weights: reverse the +1 from pre_rmsnorm_1p convention
        # "linear_attn.norm.weight" is excluded (not pre_rmsnorm_1p)
        if hf_name.endswith('norm.weight') and 'linear_attn.norm' not in hf_name:
            tensor = tensor - 1.0

        # A_log: converter stored -exp(A_log), reverse to get A_log
        if hf_name.endswith('.A_log'):
            tensor = torch.log(-tensor)

        # conv1d: converter squeezed (out, 1, kernel) → (out, kernel); unsqueeze back
        if hf_name.endswith('conv1d.weight'):
            tensor = tensor.unsqueeze(1)  # (out, kernel) → (out, 1, kernel)

        return tensor

    def _reconstruct_qwen35_fused_tensors(
        self,
        fused_components: Dict[str, Any],
        metadata: Dict[str, Any],
        as_torch: bool,
    ) -> Dict[str, Any]:
        """
        Reconstruct fused HF tensors from GGUF components for Qwen 3.5.

        Handles two fused tensor types per linear attention layer:
          1. attn_qkv + attn_gate → in_proj_qkvz  (reverses converter's QKVZ split)
          2. ssm_alpha + ssm_beta → in_proj_ba     (interleaves alpha/beta per head)

        The converter (convert_hf_to_gguf.py Qwen3NextModel.modify_tensors) does:
          in_proj_qkvz (8192,1024) → permute(1,0) → view(-1,num_k,sum_splits)
            → split into q,k,v,z → flatten each → cat([q,k,v]) as attn_qkv
            → z as attn_gate

        We reverse this: cat Q,K,V → reshape per-head → interleave with Z → flatten.
        """
        prefix = 'qwen35.'
        num_k_heads = int(metadata.get(prefix + 'ssm.group_count', 16))
        hidden_size = int(metadata.get(prefix + 'embedding_length', 1024))
        ssm_inner_size = int(metadata.get(prefix + 'ssm.inner_size', 2048))
        head_k_dim = ssm_inner_size // num_k_heads  # 128 for 0.8B
        head_v_dim = int(metadata.get(prefix + 'ssm.state_size', 128))

        result = {}

        # Group fused components by layer index
        layer_components: Dict[int, Dict[str, Any]] = {}
        for gguf_name, array in fused_components.items():
            match = re.match(r'blk\.(\d+)\.(.+)', gguf_name)
            if not match:
                continue
            layer_idx = int(match.group(1))
            component = match.group(2)
            layer_components.setdefault(layer_idx, {})[component] = array

        for layer_idx, components in sorted(layer_components.items()):
            hf_prefix = f'model.layers.{layer_idx}.linear_attn'

            # --- Reconstruct in_proj_qkvz from attn_qkv + attn_gate ---
            if 'attn_qkv.weight' in components and 'attn_gate.weight' in components:
                qkv = components['attn_qkv.weight']   # (6144, 1024)
                gate = components['attn_gate.weight']  # (2048, 1024)

                if not isinstance(qkv, np.ndarray):
                    qkv = np.array(qkv, dtype=np.float32)
                if not isinstance(gate, np.ndarray):
                    gate = np.array(gate, dtype=np.float32)

                in_proj = self._fuse_qkvz(qkv, gate, num_k_heads, head_k_dim, head_v_dim, hidden_size)

                if as_torch and HAS_TORCH:
                    in_proj = torch.from_numpy(in_proj)

                result[f'{hf_prefix}.in_proj_qkvz.weight'] = in_proj

            # --- Reconstruct in_proj_ba from ssm_beta + ssm_alpha ---
            if 'ssm_beta.weight' in components and 'ssm_alpha.weight' in components:
                beta = components['ssm_beta.weight']    # (num_v_heads, hidden)
                alpha = components['ssm_alpha.weight']  # (num_v_heads, hidden)

                if not isinstance(beta, np.ndarray):
                    beta = np.array(beta, dtype=np.float32)
                if not isinstance(alpha, np.ndarray):
                    alpha = np.array(alpha, dtype=np.float32)

                in_proj_ba = self._fuse_ba(beta, alpha, num_k_heads)

                if as_torch and HAS_TORCH:
                    in_proj_ba = torch.from_numpy(in_proj_ba)

                result[f'{hf_prefix}.in_proj_ba.weight'] = in_proj_ba

        return result

    @staticmethod
    def _fuse_qkvz(
        attn_qkv: np.ndarray,
        attn_gate: np.ndarray,
        num_k_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        hidden_size: int,
    ) -> np.ndarray:
        """
        Reverse the llama.cpp Qwen3NextModel QKVZ de-interleave.

        GGUF layout:
          attn_qkv  = [Q_all | K_all | V_all]  (q_dim + k_dim + v_dim, hidden)
          attn_gate  = [Z_all]                  (z_dim, hidden)

        HF layout (in_proj_qkvz):
          Interleaved per num_k_heads groups:
            [q_h0, k_h0, v_h0..., z_h0..., q_h1, k_h1, ...]
          Shape: (q_dim + k_dim + v_dim + z_dim, hidden)
        """
        q_dim = num_k_heads * head_k_dim
        k_dim = num_k_heads * head_k_dim

        # num_v_per_k = how many V (and Z) heads per K head group
        total_v_dim = attn_qkv.shape[0] - q_dim - k_dim
        num_v_per_k = total_v_dim // (num_k_heads * head_v_dim)
        v_per_group = num_v_per_k * head_v_dim
        z_per_group = v_per_group  # Z has same structure as V

        # Transpose to (hidden, out_dim) for grouping
        qkv_t = attn_qkv.T.copy()    # (hidden, q+k+v)
        z_t = attn_gate.T.copy()      # (hidden, z)

        # Split Q, K, V along dim-1
        q = qkv_t[:, :q_dim]           # (hidden, q_dim)
        k = qkv_t[:, q_dim:q_dim+k_dim]  # (hidden, k_dim)
        v = qkv_t[:, q_dim+k_dim:]     # (hidden, v_dim)

        # Reshape into per-K-head groups: (hidden, num_k_heads, head_dim)
        q_heads = q.reshape(hidden_size, num_k_heads, head_k_dim)
        k_heads = k.reshape(hidden_size, num_k_heads, head_k_dim)
        v_heads = v.reshape(hidden_size, num_k_heads, v_per_group)
        z_heads = z_t.reshape(hidden_size, num_k_heads, z_per_group)

        # Interleave [q, k, v, z] per group along last dim
        group_size = head_k_dim + head_k_dim + v_per_group + z_per_group
        qkvz = np.concatenate([q_heads, k_heads, v_heads, z_heads], axis=-1)
        assert qkvz.shape == (hidden_size, num_k_heads, group_size)

        # Flatten and transpose back: (hidden, total_out) → (total_out, hidden)
        total_out = num_k_heads * group_size
        result = qkvz.reshape(hidden_size, total_out).T.copy()
        return np.ascontiguousarray(result)

    @staticmethod
    def _fuse_ba(
        ssm_beta: np.ndarray,
        ssm_alpha: np.ndarray,
        num_k_heads: int,
    ) -> np.ndarray:
        """
        Reverse the llama.cpp Qwen3Next alpha/beta split.

        GGUF:  ssm_beta (num_v, hidden), ssm_alpha (num_v, hidden)
        HF:    in_proj_ba (num_v*2, hidden) interleaved: [b0,a0, b1,a1, ...]

        The interleaving groups by num_k_heads, with beta_per_group and
        alpha_per_group elements for each group.
        """
        hidden = ssm_beta.shape[-1]
        beta_per_group = ssm_beta.shape[0] // num_k_heads
        alpha_per_group = ssm_alpha.shape[0] // num_k_heads

        b = ssm_beta.reshape(num_k_heads, beta_per_group, hidden)
        a = ssm_alpha.reshape(num_k_heads, alpha_per_group, hidden)

        # Interleave: [b_group, a_group] per K-head group
        ba = np.concatenate([b, a], axis=1)  # (num_k_heads, b+a, hidden)
        result = ba.reshape(-1, hidden)
        return np.ascontiguousarray(result)

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
