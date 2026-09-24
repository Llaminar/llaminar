#!/bin/bash
# Setup ccache for faster C++ compilation

set -e

echo "🔧 Configuring ccache..."

# Keep enough expensive CUDA/HIP objects resident to reuse them across the
# Debug, Integration, and Release trees. Developers can override the project
# default without editing this script.
CCACHE_MAX_SIZE="${LLAMINAR_CCACHE_MAXSIZE:-50G}"
export CCACHE_MAXSIZE="$CCACHE_MAX_SIZE"
export CCACHE_BASEDIR="/workspaces/llaminar"
export CCACHE_NOHASHDIR=1

# Configure ccache settings.
ccache --set-config=compression=true
ccache --set-config=compression_level=6
ccache --set-config=max_size="$CCACHE_MAX_SIZE"
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

# Keep the login-shell environment current when this script is rerun in an
# existing development container.
if grep -q "^# ccache configuration$" "$HOME/.bashrc"; then
    sed -i \
        's|^export CCACHE_MAXSIZE=.*$|export CCACHE_MAXSIZE="${LLAMINAR_CCACHE_MAXSIZE:-50G}"|' \
        "$HOME/.bashrc"
    if ! grep -q "^export CCACHE_BASEDIR=" "$HOME/.bashrc"; then
        sed -i \
            '/^export CCACHE_MAXSIZE=/a export CCACHE_BASEDIR="/workspaces/llaminar"\nexport CCACHE_NOHASHDIR=1' \
            "$HOME/.bashrc"
    fi
else
    cat >> "$HOME/.bashrc" << 'EOF'

# ccache configuration
export CCACHE_DIR="$HOME/.ccache"
export CCACHE_COMPRESS=1
export CCACHE_COMPRESSLEVEL=6
export CCACHE_MAXSIZE="${LLAMINAR_CCACHE_MAXSIZE:-50G}"
export CCACHE_BASEDIR="/workspaces/llaminar"
export CCACHE_NOHASHDIR=1
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
echo "   Max size: $CCACHE_MAX_SIZE"
echo "   Compression: enabled (level 6)"
echo ""
echo "Useful commands:"
echo "   ccache-stats  - Show cache statistics"
echo "   ccache-clear  - Clear entire cache"
echo "   ccache-zero   - Zero statistics"
