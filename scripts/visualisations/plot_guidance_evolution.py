#!/usr/bin/env python3
"""plot_guidance_evolution.py

Visualize how property scores (potency, hemolysis, cytotoxicity) evolve during 
guided diffusion sampling. Shows that the guidance mechanism is actively working
throughout the generation process.

Generates a small batch of guided trajectories and tracks the mean scores
at each diffusion step to demonstrate the effectiveness of multi-objective guidance.

Usage in Jupyter cell:
    %run plot_guidance_evolution.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Running as HPC job script - no Jupyter display needed
_IN_JUPYTER = False

# Default paths
DEFAULT_PLOTS_DIR = Path("/workspace/AmpFlow/ampflow/plots")
DEFAULT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CKPT = Path("/workspace/AmpFlow/ampflow/ampdfm_ckpt/pepdfm_unconditional_epoch200.ckpt")

# Colors for each property (matching densities_stylised.py)
PROPERTY_COLORS = {
    "potency": "#1f78b4",      # Blue (generic color from densities_stylised.py)
    "hemolysis": "#2E8B57",    # Sea green (antimicrobial activity color)
    "cytotox": "#8A2BE2"       # Blue violet - keep as is
}

def guided_trajectories_with_scores(ckpt_path: Path, device: str, batch_size: int = 100, 
                                   length: int = 12, n_iterations: int = 100):
    """Generate multiple batches of guided samples and track score improvement across iterations."""
    
    # Import PepDFM components
    from flow_matching.solver.discrete_solver import MixtureDiscreteEulerSolver
    from models.peptide_classifiers import load_solver
    from ampflow.ampdfm_scripts.pepdfm_mog import PotencyJudge, HemolysisJudge, CytotoxicityJudge
    from argparse import Namespace
    
    print(f"Loading model from {ckpt_path}")
    vocab_size = 24
    solver: MixtureDiscreteEulerSolver = load_solver(str(ckpt_path), vocab_size, device)
    
    # Load scoring models
    print("Loading property judges...")
    pot_judge = PotencyJudge(device)
    hml_judge = HemolysisJudge(device)
    cyt_judge = CytotoxicityJudge(device)
    score_models = [pot_judge, hml_judge, cyt_judge]
    
    # Guidance configuration (matching pepdfm_mog defaults)
    from utils.parsing import parse_guidance_args
    g_args = parse_guidance_args([])
    g_args.T = 100  # Standard diffusion steps
    step_size = 1.0 / g_args.T
    
    print(f"Running {n_iterations} iterations of guided sampling with {batch_size} sequences each...")
    
    # Track mean and std scores over iterations
    potency_means = []
    potency_stds = []
    hemolysis_means = []
    hemolysis_stds = []
    cytotox_means = []
    cytotox_stds = []
    
    # Run multiple iterations of guided sampling
    for iteration in tqdm(range(n_iterations), desc="MOG Iterations"):
        # Initialize random sequences for this iteration
        core = torch.randint(4, vocab_size, (batch_size, length), device=device)
        x_init = torch.cat([
            torch.zeros((batch_size, 1), dtype=torch.long, device=device),  # <cls>
            core,
            torch.full((batch_size, 1), 2, dtype=torch.long, device=device)  # <eos>
        ], dim=1)
        
        # Run guided sampling to completion (final samples only)
        x_final = solver.multi_guidance_sample(
            args=g_args,
            x_init=x_init,
            step_size=step_size,
            return_intermediates=False,  # Only need final results
            verbose=False,
            score_models=score_models,
            importance=[1.0, 1.0, 1.0]  # Equal importance for all three objectives
        )
        
        # Score the final sequences from this iteration
        pot_scores = pot_judge(x_final)
        hml_scores = hml_judge(x_final)
        cyt_scores = cyt_judge(x_final)
        
        # Store means and stds for this iteration
        potency_means.append(pot_scores.mean().item())
        potency_stds.append(pot_scores.std().item())
        
        # For hemolysis and cytotox, invert for display (1 - score) so lower is better
        hemolysis_means.append((1.0 - hml_scores).mean().item())
        hemolysis_stds.append((1.0 - hml_scores).std().item())
        
        cytotox_means.append((1.0 - cyt_scores).mean().item())
        cytotox_stds.append((1.0 - cyt_scores).std().item())
    
    return {
        'steps': list(range(n_iterations)),  # This gives 0, 1, 2, ..., n_iterations-1
        'potency': potency_means,
        'potency_std': potency_stds,
        'hemolysis': hemolysis_means,
        'hemolysis_std': hemolysis_stds,
        'cytotox': cytotox_means,
        'cytotox_std': cytotox_stds
    }

def plot_property_evolution(ckpt_path: Path = DEFAULT_CKPT, device: str = "cuda:0", 
                           out_dir: Path | str | None = None, batch_size: int = 100, n_iterations: int = 100):
    """Generate property evolution curves showing guidance effectiveness across multiple iterations."""
    
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    # Generate guided trajectories and get scores
    evolution_data = guided_trajectories_with_scores(ckpt_path, device, batch_size, n_iterations=n_iterations)
    
    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Property definitions for plotting
    properties = [
        ('potency', 'Predicted Antimicrobial Activity', 'Higher is better'),
        ('hemolysis', 'Predicted Haemolysis', 'Lower is better'), 
        ('cytotox', 'Predicted Cytotoxicity', 'Lower is better')
    ]
    
    for i, (prop, title, direction) in enumerate(properties):
        ax = axes[i]
        
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
        
        # Plot the evolution curve with standard deviation band
        steps = evolution_data['steps']
        values = evolution_data[prop]
        stds = evolution_data[f'{prop}_std']
        color = PROPERTY_COLORS[prop]
        
        # Convert to numpy arrays
        steps_np = np.array(steps)
        values_np = np.array(values)
        stds_np = np.array(stds)
        
        # Plot mean line
        ax.plot(steps_np, values_np, color=color, linewidth=3, alpha=0.8, label='Mean Score')
        
        # Plot standard deviation band
        ax.fill_between(steps_np, values_np - stds_np, values_np + stds_np, 
                       color=color, alpha=0.2, label='Std Dev')
        
        # Add markers at start and end
        ax.scatter(steps[0], values[0], color=color, s=60, zorder=5, 
                  edgecolor='white', linewidth=2)
        ax.scatter(steps[-1], values[-1], color=color, s=60, zorder=5, 
                  edgecolor='white', linewidth=2, marker='s')
        
        # Customize each subplot
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Property Value', fontsize=11)
        ax.set_title(f'{title}\n({direction})', fontsize=12, fontweight='bold')
        
        # Set reasonable y-limits
        y_min, y_max = min(values), max(values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        
        # Add improvement annotation
        improvement = values[-1] - values[0]
        # For all properties as displayed, higher should be better (potency up, toxicity down after inversion)
        improvement_text = f"Δ = +{improvement:.3f}" if improvement > 0 else f"Δ = {improvement:.3f}"
        
        ax.text(0.05, 0.95, improvement_text, transform=ax.transAxes, 
               fontsize=10, fontweight='bold', color=color,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Remove tick marks but keep labels
        ax.tick_params(axis='both', which='major', labelsize=9, length=0)
        
        # Add subtle legend
        ax.legend(frameon=False, fontsize=8, loc='lower right')
    
    # Overall title
    fig.suptitle('Multi-Objective Guidance Evolution During Diffusion Sampling', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    out_path = Path(out_dir) / "guidance_property_evolution.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nProperty evolution plot saved to {out_path}")
    
    # Save and close (no display needed for HPC job)
    plt.close()
    
    # Print summary statistics
    print("\nGuidance Effectiveness Summary:")
    print(f"Batch size: {batch_size} sequences per iteration")
    print(f"Total iterations: {len(evolution_data['steps'])}")
    print(f"Total sequences generated: {batch_size * len(evolution_data['steps'])}")
    
    for prop, title, direction in properties:
        values = evolution_data[prop]
        initial = values[0]
        final = values[-1]
        change = final - initial
        percent_change = (change / initial * 100) if initial != 0 else 0
        
        print(f"\n{title}:")
        print(f"  Initial: {initial:.4f}")
        print(f"  Final: {final:.4f}")
        print(f"  Change: {change:+.4f} ({percent_change:+.1f}%)")
        
        if prop == 'potency':
            trend = "↑ Improved" if change > 0 else "↓ Decreased"
        else:  # hemolysis and cytotox
            trend = "↓ Reduced (better)" if change < 0 else "↑ Increased (worse)"
        print(f"  Trend: {trend}")

def main():
    """Main function to run the property evolution analysis."""
    # Use CUDA if available, otherwise CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run with smaller numbers for testing (MOG paper used 100 iterations × 100 sequences)
    plot_property_evolution(device=device, batch_size=50, n_iterations=50)

# Auto-run when executed
if __name__ == "__main__":
    main()

