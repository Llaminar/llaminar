"""
Unit Tests for GGUF Dimension Order Handling

This test suite locks in the correct behavior for GGUF dimension handling.

CRITICAL CONTEXT:
The GGUF file format stores tensor dimensions in REVERSE order compared to 
standard NumPy/PyTorch row-major conventions. For example:
  - GGUF file contains: [896, 151936]
  - Actual tensor shape should be: (151936, 896)

This is handled by reversing dimensions during parsing in gguf_parser.py.

The fix ensures:
1. Dimensions are reversed when reading from GGUF files
2. Tensors match expected PyTorch/NumPy shapes
3. No additional transpose is needed in the loader
4. All quantization types (F32, F16, Q4_0, Q8_0, etc.) work correctly

@author David Sanftenberg
@date 2025-10-05
"""

import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from python.reference.loaders.gguf_parser import GGUFParser
from python.reference.loaders.gguf_loader import GGUFLoader

# Try to import torch for additional validation
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestGGUFDimensionOrder(unittest.TestCase):
    """Test correct handling of GGUF dimension order."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.gguf_q4_path = Path(__file__).parent.parent.parent.parent / 'models' / 'qwen2.5-0.5b-instruct-q4_0.gguf'
        cls.gguf_fp16_path = Path(__file__).parent.parent.parent.parent / 'models' / 'qwen2.5-0.5b-instruct-fp16.gguf'
    
    def test_parser_reverses_dimensions(self):
        """Test that parser correctly reverses dimensions when reading GGUF."""
        if not self.gguf_q4_path.exists():
            self.skipTest(f"GGUF file not found: {self.gguf_q4_path}")
        
        with GGUFParser(str(self.gguf_q4_path)) as parser:
            parser.parse()
            
            # Find token embedding tensor
            token_embd = next((t for t in parser.tensors if t.name == 'token_embd.weight'), None)
            self.assertIsNotNone(token_embd, "token_embd.weight not found in GGUF")
            
            # Qwen2.5-0.5B has vocab_size=151936, hidden_size=896
            # GGUF stores this as [896, 151936] in file
            # After reversal, we should have (151936, 896)
            expected_shape = (151936, 896)
            self.assertEqual(token_embd.shape, expected_shape,
                           f"Token embedding shape mismatch. Expected {expected_shape}, got {token_embd.shape}")
            
            print(f"\n✓ Parser correctly reverses dimensions:")
            print(f"  token_embd.weight: {token_embd.shape}")
    
    def test_weight_matrix_dimensions(self):
        """Test that various weight matrices have correct dimensions after parsing."""
        if not self.gguf_q4_path.exists():
            self.skipTest(f"GGUF file not found: {self.gguf_q4_path}")
        
        with GGUFParser(str(self.gguf_q4_path)) as parser:
            parser.parse()
            
            test_cases = [
                # (tensor_name, expected_shape, description)
                ('token_embd.weight', (151936, 896), 'Embedding: vocab_size × hidden_size'),
                ('output.weight', (151936, 896), 'Output projection: vocab_size × hidden_size'),
                ('blk.0.ffn_gate.weight', (4864, 896), 'FFN gate: intermediate × hidden'),
                ('blk.0.ffn_down.weight', (896, 4864), 'FFN down: hidden × intermediate'),
                ('blk.0.attn_norm.weight', (896,), 'LayerNorm: 1D tensor (no reversal needed)'),
            ]
            
            for tensor_name, expected_shape, description in test_cases:
                with self.subTest(tensor=tensor_name):
                    tensor = next((t for t in parser.tensors if t.name == tensor_name), None)
                    self.assertIsNotNone(tensor, f"{tensor_name} not found in GGUF")
                    self.assertEqual(tensor.shape, expected_shape,
                                   f"{description}: Expected {expected_shape}, got {tensor.shape}")
                    print(f"  ✓ {tensor_name}: {tensor.shape} - {description}")
    
    def test_loader_no_additional_transpose(self):
        """Test that loader doesn't apply additional transpose after dimension reversal."""
        if not self.gguf_q4_path.exists():
            self.skipTest(f"GGUF file not found: {self.gguf_q4_path}")
        
        loader = GGUFLoader(str(self.gguf_q4_path), verbose=False)
        state_dict = loader.load_state_dict(as_torch=False, show_progress=False)
        
        # Expected shapes after loading (should match parser shapes exactly, no transpose)
        test_cases = [
            ('model.embed_tokens.weight', (151936, 896)),
            ('model.layers.0.mlp.gate_proj.weight', (4864, 896)),
            ('model.layers.0.mlp.down_proj.weight', (896, 4864)),
            ('lm_head.weight', (151936, 896)),
            ('model.layers.0.input_layernorm.weight', (896,)),
        ]
        
        for hf_name, expected_shape in test_cases:
            with self.subTest(tensor=hf_name):
                self.assertIn(hf_name, state_dict, f"{hf_name} not found in state dict")
                tensor = state_dict[hf_name]
                self.assertEqual(tensor.shape, expected_shape,
                               f"{hf_name}: Expected {expected_shape}, got {tensor.shape}")
                print(f"  ✓ {hf_name}: {tensor.shape}")
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_pytorch_model_compatibility(self):
        """Test that loaded tensors are compatible with PyTorch model expectations."""
        if not self.gguf_q4_path.exists():
            self.skipTest(f"GGUF file not found: {self.gguf_q4_path}")
        
        try:
            from transformers import AutoConfig, AutoModelForCausalLM
        except ImportError:
            self.skipTest("transformers library not available")
        
        # Load GGUF state dict
        loader = GGUFLoader(str(self.gguf_q4_path), verbose=False)
        state_dict = loader.load_state_dict(as_torch=True, show_progress=False)
        
        # Create model with same config
        config = AutoConfig.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config)
        
        # Verify shapes match
        mismatches = []
        for name, param in model.named_parameters():
            if name in state_dict:
                if param.shape != state_dict[name].shape:
                    mismatches.append(f"{name}: model={param.shape}, gguf={state_dict[name].shape}")
        
        self.assertEqual(len(mismatches), 0, 
                        f"Shape mismatches found:\n" + "\n".join(mismatches))
        
        print(f"\n✓ All tensor shapes match PyTorch model expectations")
    
    def test_fp16_dimension_consistency(self):
        """Test that FP16 and quantized models have consistent dimension handling."""
        if not self.gguf_fp16_path.exists():
            self.skipTest(f"FP16 GGUF file not found: {self.gguf_fp16_path}")
        if not self.gguf_q4_path.exists():
            self.skipTest(f"Q4_0 GGUF file not found: {self.gguf_q4_path}")
        
        # Parse both files
        with GGUFParser(str(self.gguf_fp16_path)) as fp16_parser:
            fp16_parser.parse()
            fp16_shapes = {t.name: t.shape for t in fp16_parser.tensors}
        
        with GGUFParser(str(self.gguf_q4_path)) as q4_parser:
            q4_parser.parse()
            q4_shapes = {t.name: t.shape for t in q4_parser.tensors}
        
        # Verify common tensors have same shapes
        common_tensors = set(fp16_shapes.keys()) & set(q4_shapes.keys())
        self.assertGreater(len(common_tensors), 0, "No common tensors found")
        
        mismatches = []
        for name in common_tensors:
            if fp16_shapes[name] != q4_shapes[name]:
                mismatches.append(f"{name}: FP16={fp16_shapes[name]}, Q4_0={q4_shapes[name]}")
        
        self.assertEqual(len(mismatches), 0,
                        f"Shape mismatches between FP16 and Q4_0:\n" + "\n".join(mismatches))
        
        print(f"\n✓ FP16 and Q4_0 models have consistent dimensions")
        print(f"  Verified {len(common_tensors)} common tensors")
    
    def test_embedding_values_sanity(self):
        """Test that embedding values are reasonable after dimension reversal."""
        if not self.gguf_q4_path.exists():
            self.skipTest(f"GGUF file not found: {self.gguf_q4_path}")
        
        loader = GGUFLoader(str(self.gguf_q4_path), verbose=False)
        state_dict = loader.load_state_dict(as_torch=False, show_progress=False)
        
        embed = state_dict['model.embed_tokens.weight']
        
        # Sanity checks on embedding values
        self.assertEqual(embed.shape, (151936, 896), "Embedding shape incorrect")
        self.assertFalse(np.any(np.isnan(embed)), "Embedding contains NaN values")
        self.assertFalse(np.any(np.isinf(embed)), "Embedding contains Inf values")
        
        # Check value range is reasonable for normalized embeddings
        abs_max = np.abs(embed).max()
        self.assertLess(abs_max, 10.0, f"Embedding values suspiciously large: {abs_max}")
        self.assertGreater(abs_max, 0.001, f"Embedding values suspiciously small: {abs_max}")
        
        # Check that different rows are actually different (not all zeros/same)
        row_norms = np.linalg.norm(embed, axis=1)
        self.assertGreater(np.std(row_norms), 0.01, "Embedding rows are too similar")
        
        print(f"\n✓ Embedding values pass sanity checks:")
        print(f"  Shape: {embed.shape}")
        print(f"  Value range: [{embed.min():.4f}, {embed.max():.4f}]")
        print(f"  Mean row norm: {row_norms.mean():.4f} ± {row_norms.std():.4f}")
    
    def test_inference_correctness(self):
        """
        Test that the model produces correct inference results.
        
        This is the ultimate integration test - if dimensions are wrong,
        inference will produce garbage.
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        if not self.gguf_q4_path.exists():
            self.skipTest(f"GGUF file not found: {self.gguf_q4_path}")
        
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            self.skipTest("transformers library not available")
        
        # Load model
        loader = GGUFLoader(str(self.gguf_q4_path), verbose=False)
        state_dict = loader.load_state_dict(as_torch=True, show_progress=False)
        
        config = AutoConfig.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)
        
        # Test with simple prompt
        prompt = "1+1="
        inputs = tokenizer(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_token_id = torch.argmax(probs).item()
            top_token = tokenizer.decode([top_token_id])
            top_prob = probs[top_token_id].item()
        
        # The correct answer should be '2'
        # With correct dimensions, '2' should be the top prediction
        self.assertEqual(top_token.strip(), '2',
                        f"Model should predict '2' for '1+1=', got '{top_token}' (prob={top_prob:.4f})")
        
        # Probability should be reasonably high (>0.1) if model is working correctly
        self.assertGreater(top_prob, 0.1,
                          f"Top prediction probability suspiciously low: {top_prob:.4f}")
        
        print(f"\n✓ Model produces correct inference:")
        print(f"  Prompt: '{prompt}'")
        print(f"  Top prediction: '{top_token}' (prob={top_prob:.4f})")
    
    def test_dimension_reversal_preserves_data_order(self):
        """
        Test that dimension reversal correctly handles the data layout.
        
        When GGUF stores [896, 151936], the actual data is laid out in that order.
        After reversal to (151936, 896), we should be able to reshape the flat
        data correctly and get semantically correct values.
        """
        if not self.gguf_fp16_path.exists():
            self.skipTest(f"FP16 GGUF file not found: {self.gguf_fp16_path}")
        
        # Load the tensor with our fixed loader
        loader = GGUFLoader(str(self.gguf_fp16_path), verbose=False)
        state_dict = loader.load_state_dict(as_torch=False, show_progress=False)
        embed = state_dict['model.embed_tokens.weight']
        
        # Load reference from official model
        try:
            from transformers import AutoModel
        except ImportError:
            self.skipTest("transformers library not available")
        
        try:
            reference_model = AutoModel.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)
            reference_embed = reference_model.embeddings.word_embeddings.weight.detach().cpu().numpy()
        except Exception as e:
            self.skipTest(f"Could not load reference model: {e}")
        
        # Compare shapes
        self.assertEqual(embed.shape, reference_embed.shape,
                        f"Shape mismatch: GGUF={embed.shape}, Reference={reference_embed.shape}")
        
        # Check correlation between GGUF and reference
        # Should be very high (>0.99) if dimensions are correct
        flat_gguf = embed.flatten()
        flat_ref = reference_embed.flatten()
        
        # Use a sample for correlation (full is slow)
        sample_size = min(100000, len(flat_gguf))
        correlation = np.corrcoef(flat_gguf[:sample_size], flat_ref[:sample_size])[0, 1]
        
        self.assertGreater(correlation, 0.99,
                          f"Correlation too low ({correlation:.6f}), dimensions may be wrong")
        
        print(f"\n✓ Data layout is correct:")
        print(f"  Shape: {embed.shape}")
        print(f"  Correlation with reference: {correlation:.6f}")


class TestGGUFDimensionRegressionPrevention(unittest.TestCase):
    """
    Regression tests to prevent reverting the dimension fix.
    
    These tests will FAIL if someone accidentally:
    1. Removes the dimension reversal from parser
    2. Adds back the transpose logic in loader
    3. Changes dimension reading order
    """
    
    def test_parser_has_dimension_reversal(self):
        """Verify that parser code includes dimension reversal."""
        parser_path = Path(__file__).parent.parent / 'loaders' / 'gguf_parser.py'
        self.assertTrue(parser_path.exists(), "gguf_parser.py not found")
        
        content = parser_path.read_text()
        
        # Check for dimension reversal code
        self.assertIn('dimensions.reverse()', content,
                     "REGRESSION: dimensions.reverse() missing from parser! "
                     "GGUF dimensions must be reversed during parsing.")
        
        # Check for explanatory comment
        self.assertIn('REVERSE', content.upper(),
                     "REGRESSION: Comment about dimension reversal missing!")
        
        print("\n✓ Parser contains dimension reversal code")
    
    def test_loader_no_transpose_logic(self):
        """Verify that loader doesn't have outdated transpose logic."""
        loader_path = Path(__file__).parent.parent / 'loaders' / 'gguf_loader.py'
        self.assertTrue(loader_path.exists(), "gguf_loader.py not found")
        
        content = loader_path.read_text()
        
        # The old transpose logic should NOT exist
        # (It might exist in comments/docs, but not active code)
        lines = content.split('\n')
        code_lines = [line for line in lines if not line.strip().startswith('#')]
        code_content = '\n'.join(code_lines)
        
        # Check that we don't have the old needs_transpose pattern
        self.assertNotIn('needs_transpose =', code_content,
                        "REGRESSION: Old transpose logic found in loader! "
                        "Transpose should not be applied after dimension reversal.")
        
        print("\n✓ Loader does not have outdated transpose logic")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
