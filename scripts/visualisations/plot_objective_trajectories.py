#!/usr/bin/env python3
"""plot_objective_trajectories.py

Visualize guided diffusion trajectories in 3D objective space (potency, hemolysis, cytotoxicity)
showing how sequences move through property space during multi-objective optimization.
Displays the Das-Dennis lattice structure and trajectory paths with the same beautiful
color gradient as plot_guided_trajectories.py.

This shows the TRUE optimization landscape that guides the diffusion process,
where the Das-Dennis lattice provides uniform coverage of the Pareto front.

Usage:
    python plot_objective_trajectories.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Default paths
DEFAULT_PLOTS_DIR = Path("/workspace/AmpFlow/ampflow/plots")
DEFAULT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CKPT = Path("/workspace/AmpFlow/ampflow/ampdfm_ckpt/pepdfm_unconditional_epoch200.ckpt")

def generate_objective_trajectories(ckpt_path: Path, device: str, batch_size: int = 30, 
                                  length: int = 12, steps: int = 200):
    """Generate guided trajectories and return objective scores at each timepoint."""
    
    # Import PepDFM components
    from flow_matching.solver.discrete_solver import MixtureDiscreteEulerSolver
    from models.peptide_classifiers import load_solver
    from ampflow.ampdfm_scripts.pepdfm_mog import PotencyJudge, HemolysisJudge, CytotoxicityJudge, detokenise
    from utils.parsing import parse_guidance_args
    from flow_matching.utils.multi_guidance import generate_simplex_lattice_points
    
    print(f"Loading model from {ckpt_path}")
    vocab_size = 24
    solver: MixtureDiscreteEulerSolver = load_solver(str(ckpt_path), vocab_size, device)
    
    # Load scoring models
    print("Loading property judges...")
    pot_judge = PotencyJudge(device)
    hml_judge = HemolysisJudge(device)
    cyt_judge = CytotoxicityJudge(device)
    score_models = [pot_judge, hml_judge, cyt_judge]
    
    # Guidance configuration
    g_args = parse_guidance_args([])
    g_args.T = steps
    step_size = 1.0 / g_args.T
    # Explicit time grid so solver returns intermediates at each step
    time_grid = torch.linspace(0.0, 1.0 - 1e-3, steps + 1, device=device)
    
    print(f"Generating {batch_size} guided trajectories over {steps} steps...")
    
    # Generate Das-Dennis lattice for visualization
    lattice_points = generate_simplex_lattice_points(num_obj=3, num_div=g_args.num_div)
    print(f"Das-Dennis lattice contains {lattice_points.shape[0]} weight vectors")
    
    # Initialize random sequences
    core = torch.randint(4, vocab_size, (batch_size, length), device=device)
    x_init = torch.cat([
        torch.zeros((batch_size, 1), dtype=torch.long, device=device),  # <cls>
        core,
        torch.full((batch_size, 1), 2, dtype=torch.long, device=device)  # <eos>
    ], dim=1)
    
    # Run guided sampling with intermediate states
    x_intermediates = solver.multi_guidance_sample(
        args=g_args,
        x_init=x_init,
        step_size=step_size,
        time_grid=time_grid,
        return_intermediates=True,
        verbose=False,
        score_models=score_models,
        importance=[1.0, 1.0, 1.0]
    )  # Shape: (steps+1, batch_size, sequence_length)
    
    print("Computing objective scores for all timepoints...")
    # Compute objective scores for all sequences at all timepoints
    n_timepoints, n_seqs, seq_len = x_intermediates.shape
    
    # Reshape to process all sequences at once
    all_tokens = x_intermediates.view(-1, seq_len)  # (n_timepoints * n_seqs, seq_len)
    
    with torch.no_grad():
        potency_scores = pot_judge(all_tokens).cpu().numpy()
        hemolysis_scores = hml_judge(all_tokens).cpu().numpy()  
        cytotox_scores = cyt_judge(all_tokens).cpu().numpy()
    
    # Reshape back to trajectory format and invert haemolysis and cytotoxicity scores
    potency_trajectories = potency_scores.reshape(n_timepoints, n_seqs)
    hemolysis_trajectories = 1.0 - hemolysis_scores.reshape(n_timepoints, n_seqs)  # Invert haemolysis
    cytotox_trajectories = 1.0 - cytotox_scores.reshape(n_timepoints, n_seqs)  # Invert cytotoxicity
    
    # Convert sequences for display
    print("Converting tokens to sequences...")
    trajectory_sequences = []
    for t in range(n_timepoints):
        for seq_idx in range(n_seqs):
            tokens = x_intermediates[t, seq_idx].cpu().tolist()
            sequence = detokenise(tokens)
            trajectory_sequences.append({
                'timepoint': t,
                'sequence_id': seq_idx,
                'sequence': sequence,
                'potency': potency_trajectories[t, seq_idx],
                'hemolysis': hemolysis_trajectories[t, seq_idx],  # Already inverted above
                'cytotox': cytotox_trajectories[t, seq_idx]  # Already inverted above
            })
    
    return (potency_trajectories, hemolysis_trajectories, cytotox_trajectories, 
            lattice_points, trajectory_sequences, n_timepoints, n_seqs)

def plot_objective_trajectories(ckpt_path: Path = DEFAULT_CKPT, device: str = "cuda:0", 
                              out_dir: Path | str | None = None, batch_size: int = 30, 
                              steps: int = 200, length: int = 12):
    """Generate and plot guided diffusion trajectories in 3D objective space."""
    
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    # Generate objective trajectories
    (potency_traj, hemolysis_traj, cytotox_traj, lattice_points, 
     trajectory_sequences, n_timepoints, n_seqs) = generate_objective_trajectories(
        ckpt_path, device, batch_size, length=length, steps=steps
    )
    
    # Save data for quick styling iterations
    import pickle
    data_path = Path(out_dir) / "objective_trajectories_data.pkl"
    trajectory_data = {
        'potency_traj': potency_traj,
        'hemolysis_traj': hemolysis_traj, 
        'cytotox_traj': cytotox_traj,
        'lattice_points': lattice_points,
        'trajectory_sequences': trajectory_sequences,
        'n_timepoints': n_timepoints,
        'n_seqs': n_seqs,
        'rgb_colors': ['#FF4D6D', '#D25E8A', '#A56FA7', '#7981C5', '#4C92E2', '#1FA3FF'],
        'batch_size': batch_size,
        'steps': steps,
        'length': length
    }
    with open(data_path, 'wb') as f:
        pickle.dump(trajectory_data, f)
    print(f"Trajectory data saved to {data_path}")
    
    # Create 3D plot with more whitespace
    fig = plt.figure(figsize=(14, 10))  # Wider figure for more space
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
    
    # Position axes to leave space on left for z-axis label
    ax.set_position([0.25, 0.1, 0.65, 0.8])  # [left, bottom, width, height] - move right for z-label space
    
    # Skip Das-Dennis lattice points for cleaner visualization
    lattice_np = lattice_points.numpy()
    
    # Plot trajectories with same color gradient as plot_guided_trajectories.py
    print("Plotting trajectories with color gradient...")
    
    # Define the gradient colors (same as plot_guided_trajectories.py)
    hex_colors = ['#FF4D6D', '#D25E8A', '#A56FA7', '#7981C5', '#4C92E2', '#1FA3FF']
    rgb_colors = []
    for hex_color in hex_colors:
        r = int(hex_color[1:3], 16) / 255.0
        g = int(hex_color[3:5], 16) / 255.0
        b = int(hex_color[5:7], 16) / 255.0
        rgb_colors.append((r, g, b))
    
    # Plot each trajectory
    for seq_idx in range(n_seqs):
        # Extract trajectory in objective space
        potency_path = potency_traj[:, seq_idx]
        hemolysis_path = hemolysis_traj[:, seq_idx]
        cytotox_path = cytotox_traj[:, seq_idx]
        
        # Plot trajectory segments with color gradient
        for t in range(n_timepoints - 1):
            # Progress from 0 to 1
            progress = t / (n_timepoints - 1)
            # Interpolate between the gradient colors
            scaled_progress = progress * (len(rgb_colors) - 1)
            lower_idx = int(scaled_progress)
            upper_idx = min(lower_idx + 1, len(rgb_colors) - 1)
            local_progress = scaled_progress - lower_idx
            
            # Interpolate between the two nearest colors
            r1, g1, b1 = rgb_colors[lower_idx]
            r2, g2, b2 = rgb_colors[upper_idx]
            r = r1 * (1 - local_progress) + r2 * local_progress
            g = g1 * (1 - local_progress) + g2 * local_progress
            b = b1 * (1 - local_progress) + b2 * local_progress
            
            # Plot segment
            ax.plot([potency_path[t], potency_path[t+1]], 
                   [hemolysis_path[t], hemolysis_path[t+1]], 
                   [cytotox_path[t], cytotox_path[t+1]], 
                   color=(r, g, b), linewidth=2, alpha=0.8)
        
        # Plot start and end points
        # Start point (pink/red - #FF4D6D)
        ax.scatter(potency_path[0], hemolysis_path[0], cytotox_path[0], 
                  color=rgb_colors[0], s=50, marker='o', alpha=1.0, zorder=5)
        
        # End point (blue - #1FA3FF)  
        ax.scatter(potency_path[-1], hemolysis_path[-1], cytotox_path[-1], 
                  color=rgb_colors[-1], s=50, marker='o', alpha=1.0, zorder=5)
    
    # Customize the plot
    ax.set_xlabel('Antimicrobial Activity', fontsize=12, labelpad=10)
    ax.set_ylabel('Haemolysis', fontsize=12, labelpad=10)
    # Don't set zlabel on axes - we'll add it manually
    ax.set_title('Guided Trajectories in Multi-Objective Space', 
                fontsize=14, fontweight='normal', pad=10)
    
    # Add Z-axis label manually in the left margin next to the Z-axis
    fig.text(0.18, 0.5, 'Cytotoxicity', rotation=90, 
             verticalalignment='center', horizontalalignment='center', fontsize=12)
    
    # Create custom legend with gradient line
    from matplotlib.lines import Line2D
    
    # Create multiple line segments for gradient effect in legend
    gradient_lines = []
    n_segments = len(rgb_colors) - 1
    for i in range(n_segments):
        # Create line segment with color from rgb_colors
        line_color = rgb_colors[i]
        gradient_lines.append(
            Line2D([0], [0], color=line_color, linewidth=2, alpha=0.8)
        )
    
    legend_elements = [
        Line2D([0], [0], marker='o', color=rgb_colors[0], linestyle='None',
               markersize=8, label='Start (t=0)'),
        Line2D([0], [0], marker='o', color=rgb_colors[-1], linestyle='None',
               markersize=8, label='End (t=1)'),
        # Use middle color for the flow trajectories legend line
        Line2D([0], [0], color=rgb_colors[len(rgb_colors)//2], linewidth=2, alpha=0.8,
               label='Flow Trajectories (t=0→t=1)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=False)
    
    # Set nice viewing angle - zoomed out a bit to show cytotoxicity label better
    ax.view_init(elev=15, azim=40)
    
    # Style the 3D plot
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Save individual 3D figure first
    out_path_3d = Path(out_dir) / "objective_space_trajectories_3d_only.png"
    out_path_3d.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path_3d, dpi=300, bbox_inches='tight')
    print(f"\nIndividual 3D plot saved to {out_path_3d}")
    
    # Store the 3D plot data for multi-panel figure
    fig_3d = fig
    ax_3d = ax
    
    # Create additional 2D views focusing on pairwise relationships
    print("Creating 2D pairwise trajectory plots...")
    
    # Define 2D projections for pairwise relationships (ordered for multi-panel)
    views_2d = [
        {
            "name": "antimicrobial_haemolysis", 
            "title": "Antimicrobial vs Haemolysis",
            "x_data": potency_traj, "x_label": "Antimicrobial Activity",
            "y_data": hemolysis_traj, "y_label": "Haemolysis",
            "lattice_x": lattice_np[:, 0], "lattice_y": lattice_np[:, 1]
        },
        {
            "name": "antimicrobial_cytotoxicity", 
            "title": "Antimicrobial vs Cytotoxicity",
            "x_data": potency_traj, "x_label": "Antimicrobial Activity",
            "y_data": cytotox_traj, "y_label": "Cytotoxicity",
            "lattice_x": lattice_np[:, 0], "lattice_y": lattice_np[:, 2]
        },
        {
            "name": "haemolysis_cytotoxicity", 
            "title": "Haemolysis vs Cytotoxicity",
            "x_data": hemolysis_traj, "x_label": "Haemolysis",
            "y_data": cytotox_traj, "y_label": "Cytotoxicity",
            "lattice_x": lattice_np[:, 1], "lattice_y": lattice_np[:, 2]
        }
    ]
    
    # Store 2D plot data for multi-panel figure
    views_2d_data = []
    
    for view in views_2d:
        # Create new 2D figure with same styling
        fig_2d = plt.figure(figsize=(10, 8))
        ax_2d = fig_2d.add_subplot(111)
        fig_2d.patch.set_facecolor('white')
        
        # Skip Das-Dennis lattice points for cleaner visualization
        
        # Plot trajectories with same color gradient
        for seq_idx in range(n_seqs):
            # Extract trajectory in 2D space
            x_path = view["x_data"][:, seq_idx]
            y_path = view["y_data"][:, seq_idx]
            
            # Plot trajectory segments with color gradient
            for t in range(n_timepoints - 1):
                # Progress from 0 to 1
                progress = t / (n_timepoints - 1)
                # Interpolate between the gradient colors
                scaled_progress = progress * (len(rgb_colors) - 1)
                lower_idx = int(scaled_progress)
                upper_idx = min(lower_idx + 1, len(rgb_colors) - 1)
                local_progress = scaled_progress - lower_idx
                
                # Interpolate between the two nearest colors
                r1, g1, b1 = rgb_colors[lower_idx]
                r2, g2, b2 = rgb_colors[upper_idx]
                r = r1 * (1 - local_progress) + r2 * local_progress
                g = g1 * (1 - local_progress) + g2 * local_progress
                b = b1 * (1 - local_progress) + b2 * local_progress
                
                # Plot 2D segment
                ax_2d.plot([x_path[t], x_path[t+1]], [y_path[t], y_path[t+1]], 
                          color=(r, g, b), linewidth=2, alpha=0.8)
            
            # Plot start and end points
            # Start point (pink/red - #FF4D6D)
            ax_2d.scatter(x_path[0], y_path[0], color=rgb_colors[0], s=50, 
                         marker='o', alpha=1.0, zorder=5)
            
            # End point (blue - #1FA3FF)  
            ax_2d.scatter(x_path[-1], y_path[-1], color=rgb_colors[-1], s=50, 
                         marker='o', alpha=1.0, zorder=5)
        
        # Customize the 2D plot with standard |_ axis layout
        ax_2d.set_xlabel(view["x_label"], fontsize=12)
        ax_2d.set_ylabel(view["y_label"], fontsize=12)
        ax_2d.set_title(f'{view["title"]}\n', 
                       fontsize=14, fontweight='normal', pad=10)
        
        # Apply HydrAMP-style borders (same as guided trajectories plot)
        ax_2d.spines['top'].set_visible(False)
        ax_2d.spines['right'].set_visible(False)
        ax_2d.spines['left'].set_linewidth(1.2)
        ax_2d.spines['bottom'].set_linewidth(1.2)
        ax_2d.spines['left'].set_color('#B8B8B8')
        ax_2d.spines['bottom'].set_color('#B8B8B8')
        
        # Add grid and styling
        ax_2d.grid(True, alpha=0.3, linewidth=1.2)
        ax_2d.set_axisbelow(True)
        ax_2d.tick_params(axis='both', which='major', labelsize=10, length=0)
        
        # Create custom legend with gradient line - normal position
        legend_elements = [
            Line2D([0], [0], marker='o', color=rgb_colors[0], linestyle='None',
                   markersize=6, label='Start (t=0)'),
            Line2D([0], [0], marker='o', color=rgb_colors[-1], linestyle='None',
                   markersize=6, label='End (t=1)'),
            # Use middle color for the flow trajectories legend line
            Line2D([0], [0], color=rgb_colors[len(rgb_colors)//2], linewidth=2, alpha=0.8,
                   label='Flow Trajectories (t=0→t=1)')
        ]
        # Normal legend position in upper left
        ax_2d.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=10)
        
        # Save 2D view-specific figure with smaller plot area so legend sits above without overlap
        view_path = Path(out_dir) / f"objective_space_{view['name']}.png"
        plt.subplots_adjust(left=0.15, right=0.95, top=0.65, bottom=0.15)  # Shrink plot height to leave space above for legend
        plt.savefig(view_path, dpi=300, bbox_inches='tight')
        print(f"  - {view['title']} 2D view saved to {view_path}")
        
        plt.close(fig_2d)
    
    plt.close()
    
    # Create multi-panel figure with 3 2D plots + 1 3D plot
    print("Creating multi-panel figure...")
    fig_multi = plt.figure(figsize=(20, 6))  # Wide figure for 4 panels
    fig_multi.patch.set_facecolor('white')
    
    # Create 2D subplots (3 panels on the left)
    for i, view in enumerate(views_2d):
        ax_2d = fig_multi.add_subplot(1, 4, i+1)
        
        # Skip Das-Dennis lattice points for cleaner visualization
        
        # Plot trajectories with color gradient
        for seq_idx in range(n_seqs):
            x_path = view["x_data"][:, seq_idx]
            y_path = view["y_data"][:, seq_idx]
            
            # Plot trajectory segments with color gradient
            for t in range(n_timepoints - 1):
                progress = t / (n_timepoints - 1)
                scaled_progress = progress * (len(rgb_colors) - 1)
                lower_idx = int(scaled_progress)
                upper_idx = min(lower_idx + 1, len(rgb_colors) - 1)
                local_progress = scaled_progress - lower_idx
                
                # Interpolate colors
                r1, g1, b1 = rgb_colors[lower_idx]
                r2, g2, b2 = rgb_colors[upper_idx]
                r = r1 * (1 - local_progress) + r2 * local_progress
                g = g1 * (1 - local_progress) + g2 * local_progress
                b = b1 * (1 - local_progress) + b2 * local_progress
                
                ax_2d.plot([x_path[t], x_path[t+1]], [y_path[t], y_path[t+1]], 
                          color=(r, g, b), linewidth=1.5, alpha=0.8, zorder=2)
            
            # Plot start and end points
            ax_2d.scatter(x_path[0], y_path[0], color=rgb_colors[0], s=25, 
                         marker='o', alpha=1.0, zorder=3)
            ax_2d.scatter(x_path[-1], y_path[-1], color=rgb_colors[-1], s=25, 
                         marker='o', alpha=1.0, zorder=3)
        
        # Customize subplot
        ax_2d.set_xlabel(view["x_label"], fontsize=10)
        ax_2d.set_ylabel(view["y_label"], fontsize=10)
        ax_2d.set_title(view["title"], fontsize=14, fontweight='normal', pad=10)
        
        # Apply styling
        ax_2d.spines['top'].set_visible(False)
        ax_2d.spines['right'].set_visible(False)
        ax_2d.spines['left'].set_linewidth(1.0)
        ax_2d.spines['bottom'].set_linewidth(1.0)
        ax_2d.spines['left'].set_color('#B8B8B8')
        ax_2d.spines['bottom'].set_color('#B8B8B8')
        ax_2d.grid(True, alpha=0.3, linewidth=0.8)
        ax_2d.set_axisbelow(True)
        ax_2d.tick_params(axis='both', which='major', labelsize=9, length=0)
    
    # Add 3D plot as the 4th panel (rightmost)
    ax_3d_multi = fig_multi.add_subplot(1, 4, 4, projection='3d')
    
    # Skip Das-Dennis lattice points for cleaner visualization
    
    # Plot 3D trajectories
    for seq_idx in range(n_seqs):
        potency_path = potency_traj[:, seq_idx]
        hemolysis_path = hemolysis_traj[:, seq_idx]
        cytotox_path = cytotox_traj[:, seq_idx]
        
        # Plot trajectory segments with color gradient
        for t in range(n_timepoints - 1):
            progress = t / (n_timepoints - 1)
            scaled_progress = progress * (len(rgb_colors) - 1)
            lower_idx = int(scaled_progress)
            upper_idx = min(lower_idx + 1, len(rgb_colors) - 1)
            local_progress = scaled_progress - lower_idx
            
            # Interpolate colors
            r1, g1, b1 = rgb_colors[lower_idx]
            r2, g2, b2 = rgb_colors[upper_idx]
            r = r1 * (1 - local_progress) + r2 * local_progress
            g = g1 * (1 - local_progress) + g2 * local_progress
            b = b1 * (1 - local_progress) + b2 * local_progress
            
            ax_3d_multi.plot([potency_path[t], potency_path[t+1]], 
                            [hemolysis_path[t], hemolysis_path[t+1]], 
                            [cytotox_path[t], cytotox_path[t+1]], 
                            color=(r, g, b), linewidth=1.5, alpha=0.8, zorder=2)
        
        # Plot start and end points
        ax_3d_multi.scatter(potency_path[0], hemolysis_path[0], cytotox_path[0], 
                           color=rgb_colors[0], s=25, marker='o', alpha=1.0, zorder=3)
        ax_3d_multi.scatter(potency_path[-1], hemolysis_path[-1], cytotox_path[-1], 
                           color=rgb_colors[-1], s=25, marker='o', alpha=1.0, zorder=3)
    
    # Customize 3D subplot
    ax_3d_multi.set_xlabel('Antimicrobial Activity', fontsize=9, labelpad=5)
    ax_3d_multi.set_ylabel('Haemolysis', fontsize=9, labelpad=5)
    ax_3d_multi.set_zlabel('Cytotoxicity', fontsize=9, labelpad=5)
    ax_3d_multi.set_title('3D Objective Space', fontsize=14, fontweight='normal', pad=10)
    ax_3d_multi.view_init(elev=15, azim=40)
    ax_3d_multi.grid(True, alpha=0.3)
    ax_3d_multi.tick_params(axis='both', which='major', labelsize=8)
    
    # Create shared legend below all panels
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color=rgb_colors[0], linestyle='None',
               markersize=6, label='Start (t=0)'),
        Line2D([0], [0], marker='o', color=rgb_colors[-1], linestyle='None',
               markersize=6, label='End (t=1)'),
        Line2D([0], [0], color=rgb_colors[len(rgb_colors)//2], linewidth=2, alpha=0.8,
               label='Flow Trajectories (t=0→t=1)')
    ]
    
    # Position legend below the entire figure
    legend = fig_multi.legend(handles=legend_elements, loc='lower center', 
                             bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False, fontsize=11)
    
    # Adjust layout to accommodate legend and give more space for 3D plot labels
    plt.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.15, wspace=0.20)
    
    # Add overall title
    fig_multi.suptitle('Guided Trajectories in Multi-Objective Space', 
                       fontsize=14, fontweight='normal', y=0.95)
    
    # Save multi-panel figure
    out_path_multi = Path(out_dir) / "objective_space_trajectories_multipanel.png"
    plt.savefig(out_path_multi, dpi=300, bbox_inches='tight')
    print(f"Multi-panel plot saved to {out_path_multi}")
    
    # Also save as the main trajectory plot
    out_path = Path(out_dir) / "objective_space_trajectories.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Main objective space trajectories plot saved to {out_path}")
    
    plt.close(fig_multi)
    
    # Print analysis
    print(f"\nObjective Space Analysis:")
    print(f"Das-Dennis lattice points: {lattice_points.shape[0]}")
    print(f"Number of sequences: {n_seqs}")
    print(f"Diffusion steps: {n_timepoints - 1}")
    
    # Calculate trajectory statistics in objective space
    print("\nTrajectory Movement Statistics:")
    start_potency = np.mean(potency_traj[0, :])
    end_potency = np.mean(potency_traj[-1, :])
    start_hemolysis = np.mean(hemolysis_traj[0, :])
    end_hemolysis = np.mean(hemolysis_traj[-1, :])
    start_cytotox = np.mean(cytotox_traj[0, :])
    end_cytotox = np.mean(cytotox_traj[-1, :])
    
    print(f"Antimicrobial: {start_potency:.3f} → {end_potency:.3f} (Δ = {end_potency - start_potency:+.3f})")
    print(f"Haemolysis:    {start_hemolysis:.3f} → {end_hemolysis:.3f} (Δ = {end_hemolysis - start_hemolysis:+.3f})")
    print(f"Cytotoxicity:  {start_cytotox:.3f} → {end_cytotox:.3f} (Δ = {end_cytotox - start_cytotox:+.3f})")
    
    # Calculate 3D distances moved
    objective_distances = []
    for seq_idx in range(n_seqs):
        start_point = np.array([potency_traj[0, seq_idx], hemolysis_traj[0, seq_idx], cytotox_traj[0, seq_idx]])
        end_point = np.array([potency_traj[-1, seq_idx], hemolysis_traj[-1, seq_idx], cytotox_traj[-1, seq_idx]])
        distance = np.linalg.norm(end_point - start_point)
        objective_distances.append(distance)
    
    print(f"Average distance moved in objective space: {np.mean(objective_distances):.3f} ± {np.std(objective_distances):.3f}")
    
    # Show best final sequences
    print("\nTop 5 Final Sequences by Average Score:")
    print("-" * 90)
    print(f"{'Rank':>4} {'Sequence':<15} {'Potency':>7} {'Hemolysis':>9} {'Cytotox':>8} {'Average':>8}")
    print("-" * 90)
    
    final_sequences = []
    for seq_idx in range(n_seqs):
        seq_info = trajectory_sequences[(n_timepoints-1) * n_seqs + seq_idx]
        avg_score = (seq_info['potency'] + seq_info['hemolysis'] + seq_info['cytotox']) / 3
        final_sequences.append({
            'sequence': seq_info['sequence'],
            'potency': seq_info['potency'],
            'hemolysis': seq_info['hemolysis'],
            'cytotox': seq_info['cytotox'],
            'average': avg_score
        })
    
    # Sort by average score
    final_sequences.sort(key=lambda x: x['average'], reverse=True)
    
    for rank, seq in enumerate(final_sequences[:5], 1):
        print(f"{rank:4d} {seq['sequence']:<15} {seq['potency']:7.3f} {seq['hemolysis']:9.3f} {seq['cytotox']:8.3f} {seq['average']:8.3f}")
    print("-" * 90)

def plot_from_saved_data(data_path: Path | str = None, out_dir: Path | str | None = None):
    """Plot trajectories from saved data - fast for styling iterations."""
    import pickle
    
    if data_path is None:
        data_path = DEFAULT_PLOTS_DIR / "objective_trajectories_data.pkl"
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR
        
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Saved data not found at {data_path}. Run main() first to generate data.")
    
    print(f"Loading trajectory data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract data
    potency_traj = data['potency_traj']
    hemolysis_traj = data['hemolysis_traj'] 
    cytotox_traj = data['cytotox_traj']
    lattice_points = data['lattice_points']
    trajectory_sequences = data['trajectory_sequences']
    n_timepoints = data['n_timepoints']
    n_seqs = data['n_seqs']
    rgb_colors = data['rgb_colors']
    
    lattice_np = lattice_points.numpy() if hasattr(lattice_points, 'numpy') else lattice_points
    
    print(f"Loaded data: {n_seqs} sequences, {n_timepoints} timepoints")
    
    # Create 3D plot with more whitespace (copy the plotting code from main function)
    fig = plt.figure(figsize=(14, 10))  # Wider figure for more space
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
    
    # Position axes to leave space on left for z-axis label
    ax.set_position([0.25, 0.1, 0.65, 0.8])  # [left, bottom, width, height] - move right for z-label space
    
    # Skip Das-Dennis lattice points for cleaner visualization
    
    # Plot trajectories with same color gradient
    print("Plotting trajectories with color gradient...")
    
    # Plot each trajectory
    for seq_idx in range(n_seqs):
        # Extract trajectory in objective space
        potency_path = potency_traj[:, seq_idx]
        hemolysis_path = hemolysis_traj[:, seq_idx]
        cytotox_path = cytotox_traj[:, seq_idx]
        
        # Plot trajectory segments with color gradient
        for t in range(n_timepoints - 1):
            # Progress from 0 to 1
            progress = t / (n_timepoints - 1)
            # Interpolate between the gradient colors
            scaled_progress = progress * (len(rgb_colors) - 1)
            lower_idx = int(scaled_progress)
            upper_idx = min(lower_idx + 1, len(rgb_colors) - 1)
            local_progress = scaled_progress - lower_idx
            
            # Interpolate between the two nearest colors
            r1, g1, b1 = [int(rgb_colors[lower_idx][i:i+2], 16)/255.0 for i in (1, 3, 5)]
            r2, g2, b2 = [int(rgb_colors[upper_idx][i:i+2], 16)/255.0 for i in (1, 3, 5)]
            r = r1 * (1 - local_progress) + r2 * local_progress
            g = g1 * (1 - local_progress) + g2 * local_progress
            b = b1 * (1 - local_progress) + b2 * local_progress
            
            # Plot segment
            ax.plot([potency_path[t], potency_path[t+1]], 
                   [hemolysis_path[t], hemolysis_path[t+1]], 
                   [cytotox_path[t], cytotox_path[t+1]], 
                   color=(r, g, b), linewidth=2, alpha=0.8)
        
        # Plot start and end points
        start_rgb = [int(rgb_colors[0][i:i+2], 16)/255.0 for i in (1, 3, 5)]
        end_rgb = [int(rgb_colors[-1][i:i+2], 16)/255.0 for i in (1, 3, 5)]
        
        ax.scatter(potency_path[0], hemolysis_path[0], cytotox_path[0], 
                  color=start_rgb, s=50, marker='o', alpha=1.0, zorder=5)
        ax.scatter(potency_path[-1], hemolysis_path[-1], cytotox_path[-1], 
                  color=end_rgb, s=50, marker='o', alpha=1.0, zorder=5)
    
    # Customize the plot
    ax.set_xlabel('Antimicrobial Activity', fontsize=12, labelpad=10)
    ax.set_ylabel('Haemolysis', fontsize=12, labelpad=10)
    # Don't set zlabel on axes - we'll add it manually
    ax.set_title('Guided Trajectories in Multi-Objective Space', 
                fontsize=14, fontweight='normal', pad=10)
    
    # Add Z-axis label manually in the left margin next to the Z-axis
    fig.text(0.18, 0.5, 'Cytotoxicity', rotation=90, 
             verticalalignment='center', horizontalalignment='center', fontsize=12)
    
    # Create custom legend with gradient line
    from matplotlib.lines import Line2D
    middle_rgb = [int(rgb_colors[len(rgb_colors)//2][i:i+2], 16)/255.0 for i in (1, 3, 5)]
    
    legend_elements = [
        Line2D([0], [0], marker='o', color=start_rgb, linestyle='None',
               markersize=8, label='Start (t=0)'),
        Line2D([0], [0], marker='o', color=end_rgb, linestyle='None',
               markersize=8, label='End (t=1)'),
        Line2D([0], [0], color=middle_rgb, linewidth=2, alpha=0.8,
               label='Flow Trajectories (t=0→t=1)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=False)
    
    # Set viewing angle
    ax.view_init(elev=15, azim=40)
    
    # Style the 3D plot
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Save figure - no need for subplots_adjust since we set axes position manually
    out_path = Path(out_dir) / "objective_space_trajectories.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Main plot saved to {out_path}")
    
    plt.close()

def main():
    """Main function to run the objective space trajectories analysis."""
    # Use CUDA if available, otherwise CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Generate beautiful 3D visualization
    plot_objective_trajectories(
        device=device,
        batch_size=30,   # Same as guided trajectories plot
        steps=200,       # Same step count for consistency
        length=12        # Standard AMP length
    )

# Auto-run when executed
if __name__ == "__main__":
    main()
