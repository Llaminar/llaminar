#!/usr/bin/env python3
"""
Neural Network Training for CUDA GEMM Heuristic (PyTorch + GPU Version)

This script trains a neural network using PyTorch with GPU acceleration,
then exports it to ONNX format for C++ inference via ONNX Runtime.

ADVANTAGES over scikit-learn version:
- GPU acceleration (10-50× faster training)
- Better gradient optimization (Adam with learning rate scheduling)
- More flexible architecture options
- Better handling of large datasets

Author: David Sanftenberg
Date: November 2, 2025
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Scikit-learn (for data splitting and metrics only)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def is_prime(n):
    """Helper function to check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def engineer_features(df):
    """
    Engineer features from raw benchmark data.
    
    This function adds ~86 derived features to help the neural network
    learn patterns in CUDA GEMM performance.
    """
    print("      Engineering features...")
    
    # Basic derived features
    df['threads_per_block'] = df['threads_m'] * df['threads_n']
    df['tile_size'] = df['tile_m'] * df['tile_n']
    df['tile_area'] = df['tile_m'] * df['tile_n'] * df['tile_k']
    df['work_per_thread'] = df['work_m'] * df['work_n']
    
    # Occupancy estimate (48 KB shared memory, 1024 max threads per block)
    df['smem_per_block'] = df['tile_area'] * 4  # 4 bytes per float
    df['occupancy_estimate'] = np.minimum(
        48000 / df['smem_per_block'].clip(lower=1),
        1024 / df['threads_per_block'].clip(lower=1)
    )
    
    # Arithmetic intensity (FLOPs per byte loaded)
    df['arithmetic_intensity'] = (2 * df['tile_m'] * df['tile_n'] * df['tile_k']) / \
                                  ((df['tile_m'] * df['tile_k'] + df['tile_k'] * df['tile_n']) * 4)
    
    # Problem size features
    df['m_over_tile_m'] = df['m'] / df['tile_m']
    df['n_over_tile_n'] = df['n'] / df['tile_n']
    df['k_over_tile_k'] = df['k'] / df['tile_k']
    
    # Tile coverage
    df['tile_coverage_m'] = (df['m'] % df['tile_m']) / df['tile_m']
    df['tile_coverage_n'] = (df['n'] % df['tile_n']) / df['tile_n']
    
    # Size categories
    total_size = df['m'] * df['n'] * df['k']
    df['is_tiny'] = (total_size < 1e6).astype(int)
    df['is_small'] = ((total_size >= 1e6) & (total_size < 1e7)).astype(int)
    df['is_medium'] = ((total_size >= 1e7) & (total_size < 1e8)).astype(int)
    df['is_large'] = (total_size >= 1e8).astype(int)
    
    # Shape features
    df['is_square'] = (df['n'] == df['k']).astype(int)
    df['aspect_ratio'] = df['n'] / df['k'].clip(lower=1)
    df['tile_aspect_ratio'] = df['tile_n'] / df['tile_k'].clip(lower=1)
    df['tile_shape_match'] = np.abs(df['aspect_ratio'] - df['tile_aspect_ratio'])
    
    # Batch-aware features
    df['batch_size_log2'] = np.log2(df['m'].clip(lower=1))
    df['is_single_token'] = (df['m'] == 1).astype(int)
    df['is_power_of_2_batch'] = ((df['m'] & (df['m'] - 1)) == 0).astype(int)
    
    # Alignment features
    df['n_aligned_16'] = (df['n'] % 16 == 0).astype(int)
    df['n_aligned_32'] = (df['n'] % 32 == 0).astype(int)
    df['n_aligned_64'] = (df['n'] % 64 == 0).astype(int)
    df['k_aligned_32'] = (df['k'] % 32 == 0).astype(int)
    df['n_aligned_tile'] = (df['n'] % df['tile_n'] == 0).astype(int)
    df['k_aligned_tile'] = (df['k'] % df['tile_k'] == 0).astype(int)
    
    # Efficiency features
    df['warp_efficiency'] = df['threads_per_block'] / 32
    df['blocks_per_sm_estimate'] = 48000 / df['threads_per_block'].clip(lower=32)
    df['work_imbalance'] = (df['m'] % df['tile_m']) + (df['n'] % df['tile_n'])
    df['work_total'] = df['tile_m'] * df['tile_n'] * df['tile_k']
    df['work_per_thread_normalized'] = df['work_total'] / df['threads_per_block'].clip(lower=1)
    
    # Memory/compute features
    df['bytes_loaded_per_flop'] = 1.0 / df['arithmetic_intensity'].clip(lower=0.1)
    df['prefetch_benefit'] = df['prefetch_stages'] * df['arithmetic_intensity']
    df['vec_load_aligned'] = ((df['vectorize_load'] * 4) <= 16).astype(int)
    
    # Tile coverage features
    df['m_tiles'] = np.ceil(df['m'] / df['tile_m'])
    df['n_tiles'] = np.ceil(df['n'] / df['tile_n'])
    df['k_tiles'] = np.ceil(df['k'] / df['tile_k'])
    df['partial_tiles'] = ((df['m'] % df['tile_m'] != 0).astype(int) + 
                           (df['n'] % df['tile_n'] != 0).astype(int))
    
    # Size category
    df['size_category'] = 0
    df.loc[df['is_small'] == 1, 'size_category'] = 1
    df.loc[df['is_medium'] == 1, 'size_category'] = 2
    df.loc[df['is_large'] == 1, 'size_category'] = 3
    
    # Interaction features
    df['tile_size_x_batch'] = df['tile_size'] * np.log2(df['m'].clip(lower=1))
    df['occupancy_x_intensity'] = df['occupancy_estimate'] * df['arithmetic_intensity']
    df['work_per_thread_x_batch'] = df['work_per_thread'] * df['m']
    
    # Hardware-aware features
    df['warps_per_block'] = df['threads_per_block'] / 32
    df['warp_utilization'] = (df['threads_per_block'] % 32) / 32.0
    df['smem_bank_conflicts'] = (df['tile_n'] % 32).clip(lower=1)
    df['coalescing_penalty'] = ((df['n'] % 128) != 0).astype(int)
    df['vec_load_efficiency'] = df['vectorize_load'] / 4.0
    
    # Advanced tile features
    df['tiles_per_sm'] = np.ceil(df['m_tiles'] * df['n_tiles'] / 80)
    df['tile_reuse_factor'] = df['k_tiles']
    df['tile_compute_density'] = df['tile_area'] / df['threads_per_block'].clip(lower=1)
    
    # Profiler metrics
    df['bank_conflict_risk'] = (df['tile_n'] % 32).clip(upper=16) / 16.0
    df['coalescing_score'] = (
        (df['n'] % 128 == 0).astype(float) * 1.0 +
        (df['n'] % 64 == 0).astype(float) * 0.5 +
        (df['n'] % 32 == 0).astype(float) * 0.25
    )
    df['register_pressure'] = df['work_total'] / df['threads_per_block'].clip(lower=1)
    df['mem_compute_ratio'] = df['bytes_loaded_per_flop'] * 1000
    df['warp_divergence_risk'] = (
        1.0 - ((df['threads_per_block'] & (df['threads_per_block'] - 1)) == 0).astype(float)
    )
    df['smem_kb_per_block'] = (df['smem_per_block'] / 1024.0).clip(upper=48)
    df['l1_cache_pressure'] = (df['tile_area'] * 4 / (1024 * 128)).clip(upper=1.0)
    df['occupancy_limiter'] = 0
    df.loc[df['smem_kb_per_block'] > 32, 'occupancy_limiter'] = 1
    df.loc[df['threads_per_block'] > 512, 'occupancy_limiter'] = 2
    df.loc[df['register_pressure'] > 128, 'occupancy_limiter'] = 3
    
    # ============================================================================
    # PHASE 3: Oddness and Alignment Features (Nov 2, 2025)
    # ============================================================================
    
    # Batch oddness features
    df['is_odd_batch'] = (df['m'] % 2).astype(float)
    df['is_prime_batch'] = df['m'].apply(lambda x: is_prime(int(x))).astype(float)
    df['batch_power_of_2'] = ((df['m'] & (df['m'] - 1)) == 0).astype(float)
    
    # Dimension alignment features (fine-grained)
    df['n_align_16'] = (df['n'] % 16).astype(float) / 16.0
    df['n_align_32'] = (df['n'] % 32).astype(float) / 32.0
    df['n_align_64'] = (df['n'] % 64).astype(float) / 64.0
    df['n_align_128'] = (df['n'] % 128).astype(float) / 128.0
    df['k_align_16'] = (df['k'] % 16).astype(float) / 16.0
    df['k_align_32'] = (df['k'] % 32).astype(float) / 32.0
    df['k_align_64'] = (df['k'] % 64).astype(float) / 64.0
    df['k_align_128'] = (df['k'] % 128).astype(float) / 128.0
    
    # Combined alignment scores
    df['nk_align_score_16'] = (df['n_align_16'] + df['k_align_16']) / 2.0
    df['nk_align_score_32'] = (df['n_align_32'] + df['k_align_32']) / 2.0
    df['nk_align_score_128'] = (df['n_align_128'] + df['k_align_128']) / 2.0
    
    # Total alignment quality
    df['total_alignment_quality'] = 1.0 - df['nk_align_score_128']
    
    # Oddness interaction with alignment
    df['odd_batch_aligned_dims'] = df['is_odd_batch'] * df['total_alignment_quality']
    df['aligned_batch_aligned_dims'] = (1.0 - df['is_odd_batch']) * df['total_alignment_quality']
    
    # Dimension oddness
    df['n_is_odd'] = (df['n'] % 2).astype(float)
    df['k_is_odd'] = (df['k'] % 2).astype(float)
    df['both_dims_odd'] = df['n_is_odd'] * df['k_is_odd']
    df['any_dim_odd'] = ((df['n'] % 2 != 0) | (df['k'] % 2 != 0)).astype(float)
    
    # Combined oddness score
    df['total_oddness'] = (df['is_odd_batch'] + df['n_is_odd'] + df['k_is_odd']) / 3.0
    
    # Alignment mismatch
    df['m_tile_mismatch'] = (df['m'] % df['tile_m']).astype(float) / df['tile_m']
    df['n_tile_mismatch'] = (df['n'] % df['tile_n']).astype(float) / df['tile_n']
    df['k_tile_mismatch'] = (df['k'] % df['tile_k']).astype(float) / df['tile_k']
    df['total_tile_mismatch'] = (df['m_tile_mismatch'] + df['n_tile_mismatch'] + df['k_tile_mismatch']) / 3.0
    
    return df


class GemmNet(nn.Module):
    """PyTorch neural network for GEMM performance prediction."""
    
    def __init__(self, input_size, hidden_layers=[128, 64, 32, 16], dropout=0.1):
        super(GemmNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_model(model, train_loader, val_loader, device, epochs=200, lr=0.001):
    """Train the PyTorch model."""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                      patience=10, verbose=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    print(f"\n   Training on {device}...")
    print(f"   Epochs: {epochs}, Initial LR: {lr}, Early stopping patience: {patience}")
    
    train_start = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:3d}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    train_time = time.time() - train_start
    print(f"   Training completed in {train_time:.1f}s")
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    return model


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train CUDA GEMM neural network (PyTorch + GPU)')
    parser.add_argument('--input', default='../../../../../cuda_gemm_benchmark_data.csv',
                       help='Input CSV file with benchmark data')
    parser.add_argument('--output-dir', default='../generated',
                       help='Output directory for generated files')
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[128, 64, 32, 16],
                       help='Hidden layer sizes (default: 128 64 32 16)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Training batch size (default: 256)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CUDA GEMM Neural Network Training (PyTorch + GPU)")
    print("="*80)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"\n✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠ No GPU detected, using CPU (training will be slower)")
    
    # Load data
    print(f"\n1. Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"   Loaded {len(df):,} samples")
    
    # Engineer features
    print("\n2. Engineering features...")
    df = engineer_features(df)
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in 
                   ['test_name', 'm', 'n', 'k', 'gflops',
                    'tile_m', 'tile_n', 'tile_k',
                    'threads_m', 'threads_n',
                    'work_m', 'work_n',
                    'prefetch_stages', 'vectorize_load', 'use_tensor_cores']]
    
    print(f"   Total features: {len(feature_cols)}")
    
    # Split data
    print("\n3. Splitting data...")
    
    train_tests = [
        'Qwen_0_5B_SingleToken_QKV', 'Qwen_0_5B_Batch32_QKV', 'Qwen_0_5B_FFN_Gate',
        'Qwen_1_5B_SingleToken_QKV', 'Qwen_1_5B_Batch32_QKV', 'Qwen_1_5B_FFN_Gate', 'Qwen_1_5B_FFN_Down',
        'Qwen_7B_SingleToken_QKV', 'Qwen_7B_Batch128_QKV', 'Qwen_7B_FFN_Gate',
        'Qwen_72B_SingleToken_QKV', 'Qwen_72B_Batch128_QKV', 'Qwen_72B_FFN_Down',
        'OddBatch_3x1280x1280', 'OddBatch_7x2048x2048', 'OddBatch_17x2048x2048', 'OddBatch_23x4096x4096',
        'OddDim_1x1537x2048', 'OddDim_1x2053x4096'
    ]
    
    val_tests = [
        'Qwen_4B_SingleToken_QKV', 'Qwen_4B_Batch32_QKV', 'Qwen_4B_FFN_Gate',
        'Qwen_32B_SingleToken_QKV', 'Qwen_32B_FFN_Gate'
    ]
    
    # Split by test name
    train_df = df[df['test_name'].isin(train_tests)]
    val_df = df[df['test_name'].isin(val_tests)]
    test_df = df[~df['test_name'].isin(train_tests + val_tests)]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['gflops'].values.reshape(-1, 1)
    X_val = val_df[feature_cols].values
    y_val = val_df['gflops'].values.reshape(-1, 1)
    X_test = test_df[feature_cols].values
    y_test = test_df['gflops'].values.reshape(-1, 1)
    
    print(f"      Train: {len(X_train):,} samples ({len(train_tests)} model sizes)")
    print(f"      Val:   {len(X_val):,} samples ({len(val_tests)} model sizes)")
    print(f"      Test:  {len(X_test):,} samples (unseen models)")
    
    # Scale features
    print("\n4. Scaling features (RobustScaler)...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print(f"\n5. Creating model (architecture: {args.hidden_layers})...")
    model = GemmNet(input_size=len(feature_cols), 
                   hidden_layers=args.hidden_layers,
                   dropout=args.dropout).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Train model
    print("\n6. Training model...")
    model = train_model(model, train_loader, val_loader, device, 
                       epochs=args.epochs, lr=args.lr)
    
    # Evaluate
    print("\n7. Evaluating model...")
    model.eval()
    
    with torch.no_grad():
        # Train set
        y_train_pred = model(X_train_tensor.to(device)).cpu().numpy()
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Val set
        y_val_pred = model(X_val_tensor.to(device)).cpu().numpy()
        val_r2 = r2_score(y_val, y_val_pred)
        
        # Test set
        y_test_pred = model(X_test_tensor.to(device)).cpu().numpy()
        test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n   Train R²: {train_r2:.4f}")
    print(f"   Val R²:   {val_r2:.4f}")
    print(f"   Test R²:  {test_r2:.4f}")
    
    # Export to ONNX
    print("\n8. Exporting to ONNX...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = output_dir / 'cuda_heuristic_nn.onnx'
    
    # Export model to ONNX
    dummy_input = torch.randn(1, len(feature_cols)).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=['float_input'],
        output_names=['variable'],
        dynamic_axes={'float_input': {0: 'batch_size'}, 'variable': {0: 'batch_size'}}
    )
    
    print(f"   ONNX model exported to: {onnx_path}")
    print(f"   Model size: {onnx_path.stat().st_size / 1024:.1f} KB")
    
    print("\n" + "="*80)
    print("✓ Training complete!")
    print("="*80)
    print(f"\nResults Summary:")
    print(f"  - Features: {len(feature_cols)} (99 total with oddness/alignment)")
    print(f"  - Architecture: {len(feature_cols)} → {' → '.join(map(str, args.hidden_layers))} → 1")
    print(f"  - Train R²: {train_r2:.4f}")
    print(f"  - Val R²:   {val_r2:.4f}")
    print(f"  - Test R²:  {test_r2:.4f} (unseen models)")
    print(f"  - Model: {onnx_path}")
    

if __name__ == '__main__':
    main()
