# Qwen 3.5 Community Chat Template

## Source

<https://huggingface.co/froggeric/Qwen3.5-35B-A3B-Uncensored-FernflowerAI-MLX-8bit/resolve/main/chat_template.jinja>

## License

MIT License — per the upstream community release.

Copyright (c) the Qwen team and community contributors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## Why this lives here

The chat template shipped embedded in various Qwen 3.5 GGUF artifacts has
been observed to leave the model in a degenerate repetition state after the
`</think>` block. The community-maintained template in this directory is
used as a drop-in replacement by `Qwen35GraphConfigBuilder::chatTemplateOverride()`.

The `.jinja` file is embedded into the binary at build time by CMake
(see `src/v2/models/qwen35/CMakeLists-fragment` / the custom command that
generates `Qwen35ChatTemplate.generated.h`). There is no filesystem
dependency at runtime.
