#!/usr/bin/env python3
"""
Generate CUDA GEMM factory dispatch calls for diagonal atom layout combinations.

This generates LAUNCH_TENSORCORE calls for the practical subset:
- 3 diagonal layouts: (1,1,1), (2,2,1), (4,4,1)  
- 53 tile configurations
→ 159 total calls × 2 atom types = 318 kernel instantiations

Usage: python3 generate_atom_dispatch.py > factory_code.txt
"""

# DIAGONAL ONLY: Practical subset of square atom layouts
# Filtering to diagonal avoids combinatorial explosion (9 → 3 layouts)
atom_layouts = [
    (1, 1, 1),  # Small: 1 atom → 16×8 output tile
    (2, 2, 1),  # Medium: 4 atoms → 32×16 output tile (was hardcoded default)
    (4, 4, 1),  # Large: 16 atoms → 64×32 output tile
]

# Tile sizes (from original factory)
tile_configs_k16 = [
    # Small tiles
    (16, 16, 16), (16, 32, 16), (16, 64, 16), (16, 128, 16), (16, 256, 16),
    (32, 16, 16), (32, 32, 16), (32, 64, 16), (32, 128, 16), (32, 256, 16),
    # Medium tiles
    (64, 16, 16), (64, 32, 16), (64, 64, 16), (64, 128, 16), (64, 256, 16),
    # Large tiles
    (128, 16, 16), (128, 32, 16), (128, 64, 16), (128, 128, 16), (128, 256, 16),
    (256, 16, 16), (256, 32, 16), (256, 64, 16), (256, 128, 16), (256, 256, 16),
]

tile_configs_k32 = [
    (16, 16, 32), (16, 32, 32), (16, 64, 32), (16, 128, 32),
    (32, 16, 32), (32, 32, 32), (32, 64, 32), (32, 128, 32),
    (64, 16, 32), (64, 32, 32), (64, 64, 32), (64, 128, 32),
    (128, 16, 32), (128, 32, 32), (128, 64, 32), (128, 128, 32),
]

tile_configs_k64 = [
    (16, 16, 64), (16, 32, 64), (16, 64, 64),
    (32, 16, 64), (32, 32, 64), (32, 64, 64),
    (64, 16, 64), (64, 32, 64), (64, 64, 64),
    (128, 16, 64), (128, 32, 64), (128, 64, 64),
]

def generate_dispatch_calls():
    """Generate all LAUNCH_TENSORCORE calls."""
    
    print("            // ============================================================================")
    print("            // ATOM-AWARE TILE CONFIGURATION SPACE")
    print("            // ============================================================================")
    print("            // Systematically generated for all atom layout × tile size combinations")
    print("            // Total configs: {} atom layouts × {} tile sizes = {} kernel instantiations".format(
        len(atom_layouts), 
        len(tile_configs_k16) + len(tile_configs_k32) + len(tile_configs_k64),
        len(atom_layouts) * (len(tile_configs_k16) + len(tile_configs_k32) + len(tile_configs_k64))
    ))
    print()
    
    for layout_m, layout_n, layout_k in atom_layouts:
        print(f"            // Atom layout {layout_m}×{layout_n}×{layout_k}")
        print(f"            // -------------------------------------------------------------------------")
        
        # K=16 tiles
        for tm, tn, tk in tile_configs_k16:
            print(f"            LAUNCH_TENSORCORE({layout_m}, {layout_n}, {layout_k}, {tm}, {tn}, {tk});")
        
        # K=32 tiles
        for tm, tn, tk in tile_configs_k32:
            print(f"            LAUNCH_TENSORCORE({layout_m}, {layout_n}, {layout_k}, {tm}, {tn}, {tk});")
        
        # K=64 tiles
        for tm, tn, tk in tile_configs_k64:
            print(f"            LAUNCH_TENSORCORE({layout_m}, {layout_n}, {layout_k}, {tm}, {tn}, {tk});")
        
        print()

if __name__ == "__main__":
    generate_dispatch_calls()
    
    total = len(atom_layouts) * (len(tile_configs_k16) + len(tile_configs_k32) + len(tile_configs_k64))
    print(f"// Total kernel instantiations: {total}")
    print(f"// (Each LAUNCH_TENSORCORE tries 2 atom types, so {total * 2} total branches)")
