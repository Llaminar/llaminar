# Llaminar Development Environment

This repository includes a complete VS Code dev container setup for CMake C++ development with MPI support and MCP (Model Context Protocol) servers for enhanced development experience.

## 🚀 Quick Start

1. **Prerequisites:**
   - VS Code with the "Dev Containers" extension installed
   - Docker Desktop running on your machine

2. **Open in Dev Container:**
   - Open this repository in VS Code
   - When prompted, click "Reopen in Container" or use Command Palette (`Ctrl+Shift+P`) → "Dev Containers: Reopen in Container"
   - The container will build automatically (may take a few minutes on first run)

3. **Set up MCP Server API Keys (Optional):**
   - Copy `.env.template` to `.env`
   - Fill in your API keys for web search and GitHub integration
   - Restart the dev container for changes to take effect

## 🛠️ Dev Container Features

### Development Tools
- **CMake** (latest stable version)
- **GCC/G++** compiler with C++20 and C17 support
- **Open MPI** for distributed computing
- **GDB** debugger
- **Ninja** build system
- **Git** and **GitHub CLI**
- **Node.js 20** for MCP servers

### VS Code Extensions
- C/C++ IntelliSense and debugging
- CMake Tools for project management
- GitHub Copilot integration
- Python development tools
- JSON, YAML, and Markdown support

### MCP Servers Configuration
The dev container includes pre-configured MCP servers for:
- **Web Search**: Real-time web search capabilities (Brave/Google)
- **GitHub**: Repository and issue management
- **Filesystem**: Workspace file operations
- **Memory**: Persistent context storage

## 🔧 Building and Running

### Using VS Code Tasks (Recommended)
- **Configure:** `Ctrl+Shift+P` → "Tasks: Run Task" → "cmake: configure"
- **Build:** `Ctrl+Shift+B` or "cmake: build"
- **Clean:** "cmake: clean"
- **Run with MPI:** "mpi: run (2 processes)"

### Using Terminal
```bash
# Configure CMake
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug

# Build project
cmake --build build --parallel

# Run with MPI (2 processes)
mpirun -np 2 ./build/Llaminar
```

## 🐛 Debugging

### Debug Configurations Available:
1. **Debug CMake Project**: Standard debugging of the active CMake target
2. **Debug Current File**: Debug a single C++ file
3. **MPI Debug (2 processes)**: Debug MPI applications

Use `F5` to start debugging or go to Run and Debug panel (`Ctrl+Shift+D`).

## 🌐 MCP Server Setup

### API Keys Required:
1. **Brave Search API** (Recommended):
   - Sign up at: https://api.search.brave.com/app/keys
   - Add `BRAVE_API_KEY` to your `.env` file

2. **Google Custom Search** (Alternative):
   - Get API key: https://console.developers.google.com/
   - Create CSE: https://cse.google.com/cse/
   - Add `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` to `.env`

3. **GitHub Personal Access Token**:
   - Generate at: https://github.com/settings/tokens
   - Permissions needed: `repo`, `read:org`, `read:user`
   - Add `GITHUB_PERSONAL_ACCESS_TOKEN` to `.env`

### Installing MCP Servers
The dev container automatically installs Node.js. MCP servers are installed on-demand when first used.

## 📁 Project Structure

```
llaminar/
├── .devcontainer/          # Dev container configuration
│   ├── devcontainer.json   # Container and VS Code settings
│   └── Dockerfile          # Container image definition
├── .vscode/                # VS Code workspace settings
│   ├── settings.json       # Editor and MCP configuration
│   ├── launch.json         # Debug configurations
│   └── tasks.json          # Build and run tasks
├── src/                    # Source code
│   └── main.cpp           # Example MPI application
├── CMakeLists.txt         # CMake configuration
├── .env.template          # Environment variables template
└── README.md             # This file
```

## 🎯 Features Specific to Llaminar

This setup is optimized for developing the Llaminar LLM inferencing engine:

- **MPI Integration**: Full Open MPI support for distributed computing
- **COSMA Ready**: Environment prepared for COSMA library integration
- **Modern C++**: C++20 standard for latest language features
- **Performance Debugging**: GDB with MPI debugging support
- **Scalable Architecture**: Container ready for multi-node deployment

## 🔧 Customization

### Adding Dependencies
Edit the `Dockerfile` to add system packages:
```dockerfile
RUN apt-get update && apt-get install -y \
    your-package-here \
    && rm -rf /var/lib/apt/lists/*
```

### VS Code Extensions
Add extensions to `devcontainer.json`:
```json
"extensions": [
    "your.extension.id"
]
```

### CMake Configuration
Modify `CMakeLists.txt` for project-specific needs:
- Add source files to `SOURCES`
- Link additional libraries
- Set custom compiler flags

## 🚨 Troubleshooting

### Container Build Issues
- Ensure Docker Desktop is running
- Check available disk space (>5GB recommended)
- Clear Docker cache: `docker system prune`

### MPI Permission Errors
The container is configured to allow MPI to run as the `vscode` user. If you encounter permission issues:
```bash
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
```

### MCP Server Issues
- Verify Node.js is available: `node --version`
- Check API keys in `.env` file
- Restart VS Code after changing environment variables

## 📚 Additional Resources

- [CMake Documentation](https://cmake.org/documentation/)
- [Open MPI Documentation](https://www.open-mpi.org/doc/)
- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [Model Context Protocol](https://github.com/modelcontextprotocol)

Happy coding! 🎉