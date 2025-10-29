#!/bin/bash
# Setup ccache for faster C++ compilation

set -e

echo "🔧 Configuring ccache..."

# Configure ccache settings
ccache --max-size=20G
ccache --set-config=compression=true
ccache --set-config=compression_level=6
ccache --set-config=max_size=20G
ccache --set-config=cache_dir=$HOME/.ccache

# Set up CMake to use ccache
mkdir -p "$HOME/.local/bin"

# Create ccache symlinks for compilers (CMake auto-detection)
if [ ! -L "$HOME/.local/bin/gcc" ]; then
    ln -sf /usr/bin/ccache "$HOME/.local/bin/gcc"
fi
if [ ! -L "$HOME/.local/bin/g++" ]; then
    ln -sf /usr/bin/ccache "$HOME/.local/bin/g++"
fi
if [ ! -L "$HOME/.local/bin/cc" ]; then
    ln -sf /usr/bin/ccache "$HOME/.local/bin/cc"
fi
if [ ! -L "$HOME/.local/bin/c++" ]; then
    ln -sf /usr/bin/ccache "$HOME/.local/bin/c++"
fi

# Add ccache environment variables to bashrc if not already present
if ! grep -q "CCACHE" "$HOME/.bashrc"; then
    cat >> "$HOME/.bashrc" << 'EOF'

# ccache configuration
export CCACHE_DIR="$HOME/.ccache"
export CCACHE_COMPRESS=1
export CCACHE_COMPRESSLEVEL=6
export CCACHE_MAXSIZE=20G
export PATH="$HOME/.local/bin:$PATH"

# Alias to check ccache stats
alias ccache-stats='ccache -s'
alias ccache-clear='ccache -C'
alias ccache-zero='ccache -z'
EOF
fi

# Show current ccache stats
ccache -s

echo "✅ ccache configured successfully!"
echo "   Cache location: $HOME/.ccache"
echo "   Max size: 20GB"
echo "   Compression: enabled (level 6)"
echo ""
echo "Useful commands:"
echo "   ccache-stats  - Show cache statistics"
echo "   ccache-clear  - Clear entire cache"
echo "   ccache-zero   - Zero statistics"
