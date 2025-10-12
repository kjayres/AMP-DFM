#!/usr/bin/env python3
"""plot_training_loss.py

Visualize the training and validation loss curves for AMP-DFM models
in our standard plotting style. Generates separate plots for both conditional
and unconditional models. Highlights the final best model checkpoint.

Usage in Jupyter cell:
    %run plot_training_loss.py
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Import for display in Jupyter
try:
    from IPython.display import display
    _IN_JUPYTER = True
except ImportError:
    _IN_JUPYTER = False

# Default paths
DEFAULT_PLOTS_DIR = Path.home() / "mog_dfm/ampflow/plots"
DEFAULT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONDITIONAL_CSV = Path.home() / "mog_dfm/ampflow/ampdfm_data/pepdfm_conditional_losses.csv"
DEFAULT_UNCONDITIONAL_CSV = Path.home() / "mog_dfm/ampflow/ampdfm_data/pepdfm_unconditional_losses.csv"

def parse_training_csv(csv_path: Path) -> tuple[list[int], list[float], list[float], int]:
    """Parse the training CSV file to extract epoch, train loss, val loss, and best epoch."""
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found at {csv_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract data
    epochs = df['epoch'].tolist()
    train_losses = df['train_loss'].tolist()
    val_losses = df['val_loss'].tolist()
    
    # Find the epoch with minimum validation loss
    min_val_idx = np.argmin(val_losses)
    best_epoch = epochs[min_val_idx]
    
    return epochs, train_losses, val_losses, best_epoch

def plot_training_loss(csv_path: Path, model_type: str, out_dir: Path | str | None = None):
    """Generate training and validation loss curves for a specific model type."""
    
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR
    
    print(f"Parsing {model_type} training CSV...")
    epochs, train_losses, val_losses, best_epoch = parse_training_csv(csv_path)
    
    if not epochs:
        raise ValueError(f"No training data found in {model_type} CSV file")
    
    print(f"Found {len(epochs)} training epochs for {model_type} model")
    print(f"Best {model_type} model saved at epoch {best_epoch}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Apply HydrAMP styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#B8B8B8')
    ax.spines['bottom'].set_color('#B8B8B8')
    
    # Add grid
    ax.grid(True, alpha=0.3, linewidth=1.2)
    ax.set_axisbelow(True)
    
    # Plot training and validation loss curves
    train_line = ax.plot(epochs, train_losses, color='#1f78b4', linewidth=2, 
                        label='Training Loss', alpha=0.8)
    val_line = ax.plot(epochs, val_losses, color='#ff7f0e', linewidth=2, 
                      label='Validation Loss', alpha=0.8)
    
    # Highlight the best model epoch
    if best_epoch is not None:
        best_idx = epochs.index(best_epoch)
        best_val_loss = val_losses[best_idx]
        best_train_loss = train_losses[best_idx]
        
        # Add vertical line at best epoch
        ax.axvline(x=best_epoch, color='#d62728', linestyle='--', linewidth=2, 
                  alpha=0.7, label=f'Best Model (Epoch {best_epoch})')
        
        # Add marker for best validation loss
        ax.scatter(best_epoch, best_val_loss, color='#d62728', s=100, 
                  zorder=5, edgecolor='white', linewidth=2)
        
        # Add text annotation
        ax.annotate(f'Best Val Loss: {best_val_loss:.4f}', 
                   xy=(best_epoch, best_val_loss),
                   xytext=(best_epoch + 10, best_val_loss + 0.05),
                   fontsize=10, color='#d62728', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#d62728', alpha=0.7))
    
    # Customize the plot
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'AMP-DFM ({model_type.title()}) Training Progress', fontsize=14, fontweight='normal', pad=10)
    
    # Set reasonable axis limits
    ax.set_xlim(0, max(epochs) + 5)
    
    # Add legend
    ax.legend(frameon=False, fontsize=10, loc='upper right')
    
    # Remove tick marks but keep labels
    ax.tick_params(axis='both', which='major', labelsize=10, length=0)
    
    plt.tight_layout()
    
    # Save figure
    out_path = Path(out_dir) / f"ampdfm_{model_type}_training_loss.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n{model_type.title()} training loss plot saved to {out_path}")
    
    # Display in Jupyter
    if _IN_JUPYTER:
        display(plt.gcf())
    else:
        plt.show()
    plt.close()
    
    # Print summary statistics
    print(f"\n{model_type.title()} Training Summary:")
    print(f"Initial training loss: {train_losses[0]:.4f}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Initial validation loss: {val_losses[0]:.4f}")
    print(f"Best validation loss: {min(val_losses):.4f} (Epoch {best_epoch})")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Training improvement: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
    print(f"Validation improvement: {((val_losses[0] - min(val_losses)) / val_losses[0] * 100):.1f}%")

def plot_both_models(out_dir: Path | str | None = None):
    """Generate training and validation loss curves for both conditional and unconditional models."""
    
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR
    
    print("Generating training loss plots for both models...\n")
    
    # Plot conditional model
    print("=" * 50)
    print("CONDITIONAL MODEL")
    print("=" * 50)
    plot_training_loss(DEFAULT_CONDITIONAL_CSV, "conditional", out_dir)
    
    print("\n" + "=" * 50)
    print("UNCONDITIONAL MODEL")
    print("=" * 50)
    # Plot unconditional model
    plot_training_loss(DEFAULT_UNCONDITIONAL_CSV, "unconditional", out_dir)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("Both training loss plots have been generated successfully!")

def main():
    """Main function to run the training loss visualization for both models."""
    plot_both_models()

# Auto-run when executed
if __name__ == "__main__":
    main()
elif _IN_JUPYTER:
    # Auto-run when imported or run in Jupyter
    try:
        main()
    except Exception as err:
        print("plot_training_loss –", err)
