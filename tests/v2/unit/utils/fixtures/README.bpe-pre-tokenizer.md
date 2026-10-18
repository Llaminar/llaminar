# Independent byte-BPE fixture

`bpe_pre_tokenizer.json` freezes spans and token IDs from Hugging Face
`tokenizers` 0.22.1. Its provenance records the upstream sources and complete
source SHA-256 hashes. It is a small, model-free unit fixture.

The Qwen and GPT-2 vocabularies contain their first 1,000 entries, with only
merges whose two inputs and output are in that vocabulary. This retains the
actual indentation and identifier merges that exposed cross-boundary BPE.
Golden IDs are computed by the independent reference with that reduced BPE
model and the original pre-tokenizer. They are not full-model token oracles.
Added-token dictionaries are excluded from this reduced ordinary-text oracle;
the production added-token path is exercised separately in the same test binary.
Qwen2 and Llama3 use the Qwen vocabulary projection with their own declared
Unicode expressions. Full real-GGUF parity is a separate diagnostic.

To deliberately update this fixture, authenticate the source tokenizer files,
take the described closed vocabulary/merge projection, create a fresh reference
BPE tokenizer with the original pre-tokenizer and no added tokens, and
encode each input without added special tokens. Record pre-tokenizer spans
from its returned character offsets on the raw input. Retain the declared NFC
normalizer for Qwen, and record normalized text and final IDs from `encode`.
GPT-2 and Llama3 have no normalizer. For Llama3,
use Meta's declared expression followed by ByteLevel with `add_prefix_space`
and `use_regex` both false. Never regenerate expectations during a test.

Cases cover four/eight-space Python indentation, OpenCode numbered reads,
identifiers, contractions, digit grouping, Unicode whitespace and control
characters, decomposed accents, combining scripts, astral letters, emoji
modifiers, joiners, flags, variation selectors, keycaps, CRLF and tool syntax.
The C++ test separately proves special-token boundaries, invalid UTF-8
rejection, complete byte coverage, long inputs and concurrent reuse through
the production tokenizer factory.
