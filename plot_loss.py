#!/usr/bin/env python3
"""
Script to parse training log and plot loss curves
"""
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def parse_log_file(log_path):
    """Parse the log file and extract training metrics"""
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Find all training blocks (blocks between lines of dashes)
    blocks = re.findall(r'-{30,}\n(.*?)\n-{30,}', content, re.DOTALL)
    
    data = []
    for block in blocks:
        entry = {}
        # Find all metric lines in the block
        lines = block.strip().split('\n')
        for line in lines:
            # Match lines like "| metric_name         | value    |"
            match = re.match(r'\|\s*(\w+(?:_\w+)*)\s*\|\s*([0-9.e+-]+)\s*\|', line)
            if match:
                metric_name, value_str = match.groups()
                try:
                    # Handle scientific notation
                    value = float(value_str)
                    entry[metric_name] = value
                except ValueError:
                    continue
        
        if entry:  # Only add non-empty entries
            data.append(entry)
    
    return pd.DataFrame(data)

def plot_loss_curves(df, save_dir):
    """Create comprehensive loss plots"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    # 1. Main loss curve
    if 'step' in df.columns and 'loss' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Plot main losses
        plt.subplot(2, 2, 1)
        if 'loss' in df.columns:
            plt.plot(df['step'], df['loss'], 'b-', label='Total Loss', linewidth=2)
        if 'base_loss' in df.columns:
            plt.plot(df['step'], df['base_loss'], 'r--', label='Base Loss', alpha=0.7)
        
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot loss quartiles if available
        plt.subplot(2, 2, 2)
        loss_quartiles = ['loss_q0', 'loss_q1', 'loss_q2', 'loss_q3']
        colors = ['red', 'orange', 'green', 'blue']
        for i, (q_loss, color) in enumerate(zip(loss_quartiles, colors)):
            if q_loss in df.columns:
                plt.plot(df['step'], df[q_loss], color=color, label=f'Loss Q{i}', alpha=0.7)
        
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Loss by Quartiles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot KDE losses
        plt.subplot(2, 2, 3)
        if 'kde' in df.columns:
            plt.plot(df['step'], df['kde'], 'g-', label='KDE Loss', linewidth=2)
        if 'kde_weighted' in df.columns:
            plt.plot(df['step'], df['kde_weighted'], 'g--', label='KDE Weighted', alpha=0.7)
        if 'kde_2nd' in df.columns:
            plt.plot(df['step'], df['kde_2nd'], 'm-', label='KDE 2nd Order', alpha=0.7)
        if 'kde_2nd_weighted' in df.columns:
            plt.plot(df['step'], df['kde_2nd_weighted'], 'm--', label='KDE 2nd Weighted', alpha=0.7)
        
        plt.xlabel('Training Step')
        plt.ylabel('KDE Loss')
        plt.title('KDE-based Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot gradient norm and loss scale
        plt.subplot(2, 2, 4)
        if 'grad_norm' in df.columns:
            plt.plot(df['step'], df['grad_norm'], 'purple', label='Gradient Norm', alpha=0.7)
        if 'lg_loss_scale' in df.columns:
            plt.plot(df['step'], df['lg_loss_scale'], 'brown', label='Log Loss Scale', alpha=0.7)
        if 'param_norm' in df.columns:
            # Normalize param_norm for better visualization
            normalized_param_norm = df['param_norm'] / df['param_norm'].max() * df['grad_norm'].max()
            plt.plot(df['step'], normalized_param_norm, 'orange', label='Param Norm (normalized)', alpha=0.7)
        
        plt.xlabel('Training Step')
        plt.ylabel('Value')
        plt.title('Training Diagnostics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_loss_overview.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'training_loss_overview.pdf', bbox_inches='tight')
        print(f"Saved overview plot to {save_dir}/training_loss_overview.png")
        
    # 2. Detailed loss progression
    plt.figure(figsize=(15, 10))
    
    # Main loss with smoothed version
    plt.subplot(3, 2, 1)
    if 'loss' in df.columns:
        plt.plot(df['step'], df['loss'], 'b-', alpha=0.6, label='Loss (raw)')
        # Add smoothed version using rolling mean
        if len(df) > 10:
            window_size = max(1, len(df) // 50)  # Adaptive window size
            smoothed = df['loss'].rolling(window=window_size, center=True).mean()
            plt.plot(df['step'], smoothed, 'b-', linewidth=3, label='Loss (smoothed)')
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Main Loss Progression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Base loss components
    plt.subplot(3, 2, 2)
    base_loss_cols = [col for col in df.columns if col.startswith('base_loss')]
    for col in base_loss_cols:
        plt.plot(df['step'], df[col], label=col, alpha=0.7)
    
    plt.xlabel('Training Step')
    plt.ylabel('Base Loss')
    plt.title('Base Loss Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # KDE losses detailed
    plt.subplot(3, 2, 3)
    kde_cols = [col for col in df.columns if col.startswith('kde') and not col.endswith(('q0', 'q1', 'q2', 'q3'))]
    for col in kde_cols:
        plt.plot(df['step'], df[col], label=col, alpha=0.7)
    
    plt.xlabel('Training Step')
    plt.ylabel('KDE Loss')
    plt.title('KDE Loss Details')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # KDE quartiles
    plt.subplot(3, 2, 4)
    kde_quartile_cols = [col for col in df.columns if col.startswith('kde_') and col.endswith(('q0', 'q1', 'q2', 'q3'))]
    colors = plt.cm.Set1(np.linspace(0, 1, len(kde_quartile_cols)))
    for col, color in zip(kde_quartile_cols, colors):
        plt.plot(df['step'], df[col], label=col, color=color, alpha=0.7)
    
    plt.xlabel('Training Step')
    plt.ylabel('KDE Loss')
    plt.title('KDE Loss by Quartiles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Training diagnostics
    plt.subplot(3, 2, 5)
    if 'grad_norm' in df.columns:
        plt.plot(df['step'], df['grad_norm'], 'purple', label='Gradient Norm', linewidth=2)
    
    plt.xlabel('Training Step')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm Progression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Loss scale
    plt.subplot(3, 2, 6)
    if 'lg_loss_scale' in df.columns:
        plt.plot(df['step'], df['lg_loss_scale'], 'brown', label='Log Loss Scale', linewidth=2)
    
    plt.xlabel('Training Step')
    plt.ylabel('Log Loss Scale')
    plt.title('Loss Scale Progression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_loss_detailed.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'training_loss_detailed.pdf', bbox_inches='tight')
    print(f"Saved detailed plot to {save_dir}/training_loss_detailed.png")
    
    # 3. Create a simple loss-only plot
    plt.figure(figsize=(10, 6))
    if 'loss' in df.columns:
        plt.plot(df['step'], df['loss'], 'b-', linewidth=2, label='Total Loss')
        
        # Add smoothed version
        if len(df) > 10:
            window_size = max(1, len(df) // 50)
            smoothed = df['loss'].rolling(window=window_size, center=True).mean()
            plt.plot(df['step'], smoothed, 'r-', linewidth=3, label='Smoothed Loss')
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Progression', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'simple_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'simple_loss_curve.pdf', bbox_inches='tight')
    print(f"Saved simple loss curve to {save_dir}/simple_loss_curve.png")
    
    return df

def print_training_summary(df):
    """Print a summary of the training progress"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    if 'step' in df.columns:
        print(f"Total training steps: {df['step'].iloc[-1]:,.0f}")
        print(f"Training entries logged: {len(df):,}")
    
    if 'loss' in df.columns:
        initial_loss = df['loss'].iloc[0]
        final_loss = df['loss'].iloc[-1]
        min_loss = df['loss'].min()
        print(f"\nLoss progression:")
        print(f"  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Minimum loss: {min_loss:.4f}")
        print(f"  Loss reduction: {((initial_loss - final_loss) / initial_loss * 100):.1f}%")
    
    if 'base_loss' in df.columns:
        initial_base = df['base_loss'].iloc[0]
        final_base = df['base_loss'].iloc[-1]
        print(f"\nBase loss progression:")
        print(f"  Initial base loss: {initial_base:.4f}")
        print(f"  Final base loss: {final_base:.4f}")
        print(f"  Base loss reduction: {((initial_base - final_base) / initial_base * 100):.1f}%")
    
    if 'kde' in df.columns:
        initial_kde = df['kde'].iloc[0]
        final_kde = df['kde'].iloc[-1]
        print(f"\nKDE loss progression:")
        print(f"  Initial KDE loss: {initial_kde:.4f}")
        print(f"  Final KDE loss: {final_kde:.4f}")
        print(f"  KDE loss reduction: {((initial_kde - final_kde) / initial_kde * 100):.1f}%")

def main():
    log_path = "/home/qiudihe/rad/CM_MRI/output_logs/run_2nd_order_histogram_attn_Aug17/log.txt"
    save_dir = "/home/qiudihe/rad/CM_MRI/output_logs/run_2nd_order_histogram_attn_Aug17/loss_plots"
    
    print(f"Parsing log file: {log_path}")
    df = parse_log_file(log_path)
    
    if df.empty:
        print("ERROR: No data could be extracted from the log file!")
        return
    
    print(f"Successfully extracted {len(df)} training entries")
    print(f"Available metrics: {list(df.columns)}")
    
    # Print training summary
    print_training_summary(df)
    
    # Create plots
    print(f"\nCreating plots and saving to: {save_dir}")
    plot_loss_curves(df, save_dir)
    
    # Save the extracted data as CSV for future reference
    csv_path = f"{save_dir}/training_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved training data to: {csv_path}")
    
    print("\nDone! Check the plots in the loss_plots directory.")

if __name__ == "__main__":
    main()
