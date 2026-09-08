"""
Unit Tests for GGUF Parser

Tests for GGUF file format parsing including:
- Header parsing (version, tensor count, metadata count)
- Metadata extraction (key-value pairs, model config)
- Tensor info parsing (name, shape, type, offset)
- Real file integration tests

@author David Sanftenberg
"""

import unittest
import tempfile
import struct
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from python.reference.loaders.gguf_parser import (
    GGUFParser,
    GGUFValueType,
    GGUFTensorType,
)


def _write_gguf_string(handle, value: str) -> None:
    """Write one GGUF length-prefixed UTF-8 string."""
    encoded = value.encode("utf-8")
    handle.write(struct.pack("<Q", len(encoded)))
    handle.write(encoded)


def _write_synthetic_split_shard(
    path: Path,
    *,
    split_no: int,
    split_count: int,
    tensor_name: str,
    tensor_shape: tuple[int, ...],
    tensor_values: tuple[float, ...],
) -> None:
    """Write a minimal standards-shaped split GGUF shard for parser tests."""
    metadata = (
        ("general.architecture", GGUFValueType.STRING, "qwen35moe"),
        ("qwen35moe.block_count", GGUFValueType.UINT32, 2),
        ("qwen35moe.nextn_predict_layers", GGUFValueType.UINT32, 1),
        ("split.no", GGUFValueType.UINT16, split_no),
        ("split.count", GGUFValueType.UINT16, split_count),
        ("split.tensors.count", GGUFValueType.INT32, split_count),
    )
    with path.open("wb") as handle:
        handle.write(b"GGUF")
        handle.write(struct.pack("<IQQ", 3, 1, len(metadata)))
        for key, value_type, value in metadata:
            _write_gguf_string(handle, key)
            handle.write(struct.pack("<I", int(value_type)))
            if value_type == GGUFValueType.STRING:
                _write_gguf_string(handle, value)
            elif value_type == GGUFValueType.UINT16:
                handle.write(struct.pack("<H", value))
            elif value_type == GGUFValueType.UINT32:
                handle.write(struct.pack("<I", value))
            elif value_type == GGUFValueType.INT32:
                handle.write(struct.pack("<i", value))
            else:  # pragma: no cover - the fixture owns the metadata schema.
                raise AssertionError(f"unsupported synthetic type: {value_type}")

        _write_gguf_string(handle, tensor_name)
        handle.write(struct.pack("<I", len(tensor_shape)))
        # GGUF persists dimensions in the reverse of NumPy/PyTorch order.
        for dimension in reversed(tensor_shape):
            handle.write(struct.pack("<Q", dimension))
        handle.write(struct.pack("<IQ", int(GGUFTensorType.F32), 0))

        padding = (-handle.tell()) % GGUFParser.ALIGNMENT
        handle.write(b"\0" * padding)
        handle.write(struct.pack(f"<{len(tensor_values)}f", *tensor_values))


class TestGGUFSplitSet(unittest.TestCase):
    """Lock down aggregate inventory and shard-local byte ownership."""

    def _write_pair(self, directory: Path) -> tuple[Path, Path]:
        first = directory / "tiny-00001-of-00002.gguf"
        second = directory / "tiny-00002-of-00002.gguf"
        _write_synthetic_split_shard(
            first,
            split_no=0,
            split_count=2,
            tensor_name="output_norm.weight",
            tensor_shape=(2,),
            tensor_values=(1.25, -2.5),
        )
        _write_synthetic_split_shard(
            second,
            split_no=1,
            split_count=2,
            tensor_name="blk.1.nextn.eh_proj.weight",
            tensor_shape=(2, 2),
            tensor_values=(3.0, 4.0, 5.0, 6.0),
        )
        return first, second

    def test_split_root_exposes_all_tensors_and_routes_reads(self):
        """A metadata-only-style first shard must not hide later tensors."""
        with tempfile.TemporaryDirectory() as raw_directory:
            first, _ = self._write_pair(Path(raw_directory))
            with GGUFParser(first) as parser:
                parser.parse()
                self.assertEqual(parser.tensor_count, 2)
                self.assertEqual(
                    [tensor.name for tensor in parser.tensors],
                    ["output_norm.weight", "blk.1.nextn.eh_proj.weight"],
                )
                self.assertEqual(parser.get_config_dict()["num_hidden_layers"], 1)
                sidecar = parser.tensors[1]
                values = struct.unpack("<4f", parser.read_tensor_data(sidecar))
                self.assertEqual(values, (3.0, 4.0, 5.0, 6.0))

    def test_split_inventory_is_deterministic_from_any_shard(self):
        """Opening shard two still produces canonical shard-order iteration."""
        with tempfile.TemporaryDirectory() as raw_directory:
            _, second = self._write_pair(Path(raw_directory))
            with GGUFParser(second) as parser:
                parser.parse()
                self.assertEqual(
                    [tensor.name for tensor in parser.tensors],
                    ["output_norm.weight", "blk.1.nextn.eh_proj.weight"],
                )

    def test_missing_split_fails_closed(self):
        """A declared split set cannot silently degrade to one shard."""
        with tempfile.TemporaryDirectory() as raw_directory:
            first = Path(raw_directory) / "tiny-00001-of-00002.gguf"
            _write_synthetic_split_shard(
                first,
                split_no=0,
                split_count=2,
                tensor_name="output_norm.weight",
                tensor_shape=(1,),
                tensor_values=(1.0,),
            )
            with GGUFParser(first) as parser:
                with self.assertRaisesRegex(FileNotFoundError, "missing shard"):
                    parser.parse()


class TestGGUFRealFile(unittest.TestCase):
    """Test parsing real GGUF files (integration tests)."""
    
    def test_qwen_gguf_file(self):
        """Test parsing real Qwen GGUF file if available."""
        gguf_path = Path(__file__).parent.parent.parent.parent / 'models' / 'qwen2.5-0.5b-instruct-q4_0.gguf'
        
        if not gguf_path.exists():
            self.skipTest(f"GGUF file not found: {gguf_path}")
        
        parser = GGUFParser(str(gguf_path))
        parser.parse()
        
        # Verify header was parsed
        self.assertEqual(parser.version, 3)
        self.assertIsNotNone(parser.tensor_count)
        self.assertIsNotNone(parser.metadata_kv_count)
        
        # Verify metadata was parsed
        self.assertIn('general.architecture', parser.metadata)
        self.assertEqual(parser.metadata['general.architecture'], 'qwen2')
        
        # Verify tensors were parsed
        self.assertEqual(len(parser.tensors), parser.tensor_count)
        
        # Verify some expected tensors exist
        tensor_names = [t.name for t in parser.tensors]
        self.assertIn('token_embd.weight', tensor_names)
        
        # Verify config extraction
        config_dict = parser.get_config_dict()
        self.assertIn('hidden_size', config_dict)
        self.assertIn('num_hidden_layers', config_dict)
        self.assertIn('num_attention_heads', config_dict)
        
        # Verify expected Qwen config values
        self.assertEqual(config_dict['hidden_size'], 896)
        self.assertEqual(config_dict['num_hidden_layers'], 24)
        
        # Verify model type
        self.assertEqual(parser.get_model_type(), 'qwen2')
        
        print(f"\n✓ Qwen GGUF parsed successfully")
        print(f"  - Version: {parser.version}")
        print(f"  - Tensors: {parser.tensor_count}")
        print(f"  - Metadata KVs: {parser.metadata_kv_count}")
        print(f"  - Hidden size: {config_dict['hidden_size']}")
        print(f"  - Layers: {config_dict['num_hidden_layers']}")
    
    def test_llama_gguf_file(self):
        """Test parsing real LLaMA GGUF file if available."""
        gguf_path = Path(__file__).parent.parent.parent.parent / 'models' / 'Llama-3.2-1B-Instruct-Q4_0.gguf'
        
        if not gguf_path.exists():
            self.skipTest(f"GGUF file not found: {gguf_path}")
        
        parser = GGUFParser(str(gguf_path))
        parser.parse()
        
        # Verify header was parsed
        self.assertEqual(parser.version, 3)
        self.assertIsNotNone(parser.tensor_count)
        self.assertIsNotNone(parser.metadata_kv_count)
        
        # Verify metadata was parsed
        self.assertIn('general.architecture', parser.metadata)
        self.assertEqual(parser.metadata['general.architecture'], 'llama')
        
        # Verify tensors were parsed
        self.assertEqual(len(parser.tensors), parser.tensor_count)
        
        # Verify some expected tensors exist
        tensor_names = [t.name for t in parser.tensors]
        self.assertIn('token_embd.weight', tensor_names)
        
        # Verify config extraction
        config_dict = parser.get_config_dict()
        self.assertIn('hidden_size', config_dict)
        self.assertIn('num_hidden_layers', config_dict)
        self.assertIn('num_attention_heads', config_dict)
        
        # Verify model type
        self.assertEqual(parser.get_model_type(), 'llama')
        
        print(f"\n✓ LLaMA GGUF parsed successfully")
        print(f"  - Version: {parser.version}")
        print(f"  - Tensors: {parser.tensor_count}")
        print(f"  - Metadata KVs: {parser.metadata_kv_count}")
        print(f"  - Hidden size: {config_dict['hidden_size']}")
        print(f"  - Layers: {config_dict['num_hidden_layers']}")


class TestGGUFTensorInfo(unittest.TestCase):
    """Test tensor info properties from real files."""
    
    def test_qwen_tensor_shapes(self):
        """Test that tensor shapes are correctly parsed."""
        gguf_path = Path(__file__).parent.parent.parent.parent / 'models' / 'qwen2.5-0.5b-instruct-q4_0.gguf'
        
        if not gguf_path.exists():
            self.skipTest(f"GGUF file not found: {gguf_path}")
        
        parser = GGUFParser(str(gguf_path))
        parser.parse()
        
        # Find embedding tensor
        embedding_tensor = None
        for tensor in parser.tensors:
            if tensor.name == 'token_embd.weight':
                embedding_tensor = tensor
                break
        
        self.assertIsNotNone(embedding_tensor, "Embedding tensor not found")
        
        # Verify tensor properties
        self.assertIsInstance(embedding_tensor.shape, tuple)
        self.assertEqual(len(embedding_tensor.shape), 2, "Embedding should be 2D")
        self.assertGreater(embedding_tensor.n_elements, 0)
        
        # Verify quantization detection
        self.assertTrue(embedding_tensor.is_quantized, "Q4_0 should be detected as quantized")
        
        print(f"\n✓ Tensor properties validated")
        print(f"  - Embedding shape: {embedding_tensor.shape}")
        print(f"  - Elements: {embedding_tensor.n_elements:,}")
        print(f"  - Type: {embedding_tensor.type.name}")
        print(f"  - Quantized: {embedding_tensor.is_quantized}")


class TestGGUFMetadataExtraction(unittest.TestCase):
    """Test metadata extraction and config mapping."""
    
    def test_qwen_metadata_keys(self):
        """Test that expected Qwen metadata keys are present."""
        gguf_path = Path(__file__).parent.parent.parent.parent / 'models' / 'qwen2.5-0.5b-instruct-q4_0.gguf'
        
        if not gguf_path.exists():
            self.skipTest(f"GGUF file not found: {gguf_path}")
        
        parser = GGUFParser(str(gguf_path))
        parser.parse()
        
        # Check for expected metadata keys
        expected_keys = [
            'general.architecture',
            'qwen2.context_length',
            'qwen2.embedding_length',
            'qwen2.block_count',
            'qwen2.attention.head_count',
        ]
        
        for key in expected_keys:
            self.assertIn(key, parser.metadata, f"Missing metadata key: {key}")
        
        # Verify metadata types
        self.assertIsInstance(parser.metadata['general.architecture'], str)
        self.assertIsInstance(parser.metadata['qwen2.context_length'], int)
        self.assertIsInstance(parser.metadata['qwen2.embedding_length'], int)
        
        print(f"\n✓ Metadata keys validated")
        print(f"  - Architecture: {parser.metadata['general.architecture']}")
        print(f"  - Context length: {parser.metadata['qwen2.context_length']}")
        print(f"  - Embedding length: {parser.metadata['qwen2.embedding_length']}")
    
    def test_config_dict_mapping(self):
        """Test that config dict correctly maps GGUF metadata to HuggingFace format."""
        gguf_path = Path(__file__).parent.parent.parent.parent / 'models' / 'qwen2.5-0.5b-instruct-q4_0.gguf'
        
        if not gguf_path.exists():
            self.skipTest(f"GGUF file not found: {gguf_path}")
        
        parser = GGUFParser(str(gguf_path))
        parser.parse()
        
        config_dict = parser.get_config_dict()
        
        # Verify mapping from GGUF to HuggingFace keys
        # qwen2.embedding_length -> hidden_size
        self.assertEqual(
            config_dict['hidden_size'],
            parser.metadata['qwen2.embedding_length']
        )
        
        # qwen2.block_count -> num_hidden_layers
        self.assertEqual(
            config_dict['num_hidden_layers'],
            parser.metadata['qwen2.block_count']
        )
        
        # qwen2.attention.head_count -> num_attention_heads
        self.assertEqual(
            config_dict['num_attention_heads'],
            parser.metadata['qwen2.attention.head_count']
        )
        
        print(f"\n✓ Config dict mapping validated")
        print(f"  - GGUF qwen2.embedding_length → HF hidden_size")
        print(f"  - GGUF qwen2.block_count → HF num_hidden_layers")
        print(f"  - GGUF qwen2.attention.head_count → HF num_attention_heads")


class TestGGUFTensorTypes(unittest.TestCase):
    """Test tensor type detection and properties."""
    
    def test_quantization_detection(self):
        """Test that different tensor types are correctly identified."""
        gguf_path = Path(__file__).parent.parent.parent.parent / 'models' / 'qwen2.5-0.5b-instruct-q4_0.gguf'
        
        if not gguf_path.exists():
            self.skipTest(f"GGUF file not found: {gguf_path}")
        
        parser = GGUFParser(str(gguf_path))
        parser.parse()
        
        # Collect tensor types
        tensor_types = {}
        for tensor in parser.tensors:
            type_name = tensor.type.name
            tensor_types[type_name] = tensor_types.get(type_name, 0) + 1
        
        # Verify we have quantized tensors
        self.assertTrue(any('Q' in t for t in tensor_types.keys()), 
                       "Expected to find quantized tensors")
        
        # Verify quantization detection works
        for tensor in parser.tensors:
            if tensor.type in (GGUFTensorType.F32, GGUFTensorType.F16):
                self.assertFalse(tensor.is_quantized)
            else:
                self.assertTrue(tensor.is_quantized)
        
        print(f"\n✓ Tensor type distribution:")
        for type_name, count in sorted(tensor_types.items()):
            print(f"  - {type_name}: {count} tensors")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
