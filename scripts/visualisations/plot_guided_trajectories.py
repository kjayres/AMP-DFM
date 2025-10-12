#!/usr/bin/env python3
"""plot_guided_trajectories.py

Visualize guided diffusion trajectories showing how individual sequences evolve
through chemical space during multi-objective sampling. Uses UMAP to project
ESM-2 embeddings into 2D space and shows the path each sequence takes from
random initialization to optimized final state.

Generates trajectories for a small batch and shows their evolution paths
overlaid on a scatter plot, demonstrating how guidance steers sequences
towards desired regions of chemical space.

Usage as HPC job:
    qsub plot_guided_trajectories.sh
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

def esm_embed_sequences(sequences: list[str], device: str, batch_size: int = 32) -> np.ndarray:
    """Embed sequences using ESM-2 with batching for memory efficiency."""
    from transformers import AutoTokenizer, EsmModel
    
    # Load ESM-2 model
    print("Loading ESM-2 model...")
    esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
    esm_model.eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    
    embeddings = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(sequences), batch_size), desc="Embedding sequences"):
        batch_seqs = sequences[i:i+batch_size]
        
        # Tokenize batch
        encoded = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = esm_model(**encoded)
            # Mean pool over sequence length
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            masked_embeddings = outputs.last_hidden_state * attention_mask
            pooled = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
            embeddings.append(pooled.cpu().numpy())
    
    return np.vstack(embeddings)

def generate_guided_trajectories(ckpt_path: Path, device: str, batch_size: int = 10, 
                                length: int = 12, steps: int = 50):
    """Generate guided trajectories and return sequences at each timepoint."""
    
    # Import PepDFM components
    from flow_matching.solver.discrete_solver import MixtureDiscreteEulerSolver
    from models.peptide_classifiers import load_solver
    from ampflow.ampdfm_scripts.pepdfm_mog import PotencyJudge, HemolysisJudge, CytotoxicityJudge, detokenise
    # Use the same guidance arg schema as pepdfm_mog for consistency
    from utils.parsing import parse_guidance_args
    
    print(f"Loading model from {ckpt_path}")
    vocab_size = 24
    solver: MixtureDiscreteEulerSolver = load_solver(str(ckpt_path), vocab_size, device)
    
    # Load scoring models
    print("Loading property judges...")
    pot_judge = PotencyJudge(device)
    hml_judge = HemolysisJudge(device)
    cyt_judge = CytotoxicityJudge(device)
    score_models = [pot_judge, hml_judge, cyt_judge]
    
    # Guidance configuration (reduced steps for faster execution)
    # Parse with defaults from utils.parsing, then override T for our step count
    g_args = parse_guidance_args([])
    g_args.T = steps
    step_size = 1.0 / g_args.T
    # Explicit time grid so solver returns intermediates at each step
    time_grid = torch.linspace(0.0, 1.0 - 1e-3, steps + 1, device=device)
    
    print(f"Generating {batch_size} guided trajectories over {steps} steps...")
    
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
    
    print("Converting tokens to sequences...")
    # Convert all timepoints to sequences
    all_sequences = []
    trajectory_info = []
    
    n_timepoints, n_seqs, seq_len = x_intermediates.shape
    
    for t in range(n_timepoints):
        for seq_idx in range(n_seqs):
            tokens = x_intermediates[t, seq_idx].cpu().tolist()
            sequence = detokenise(tokens)
            all_sequences.append(sequence)
            trajectory_info.append({
                'timepoint': t,
                'sequence_id': seq_idx,
                'sequence': sequence
            })
    
    return all_sequences, trajectory_info, n_timepoints, n_seqs, x_intermediates, pot_judge, hml_judge, cyt_judge

def plot_guided_trajectories(ckpt_path: Path = DEFAULT_CKPT, device: str = "cuda:0", 
                           out_dir: Path | str | None = None, batch_size: int = 8, steps: int = 50,
                           length: int = 12):
    """Generate and plot guided diffusion trajectories in UMAP space."""
    
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    # Generate guided trajectories
    all_sequences, trajectory_info, n_timepoints, n_seqs, x_intermediates, pot_judge, hml_judge, cyt_judge = generate_guided_trajectories(
        ckpt_path, device, batch_size, length=length, steps=steps
    )
    
    # Embed all sequences using ESM-2
    print("Computing ESM-2 embeddings...")
    embeddings = esm_embed_sequences(all_sequences, device)
    
    # Compute UMAP projection
    print("Computing UMAP projection...")
    import umap
    
    # Generate background sequences for context
    print("Generating background sequences...")
    from ampflow.ampdfm_scripts.pepdfm_mog import detokenise
    n_background = 1000
    vocab_size = 24  # PepDFM vocabulary size
    background_seqs = []
    for _ in range(n_background):
        tokens = torch.randint(4, vocab_size, (1, length), device=device)
        seq = detokenise(tokens[0].tolist())
        background_seqs.append(seq)
    
    # Embed background sequences
    background_emb = esm_embed_sequences(background_seqs, device)
    
    # Combine background and trajectory embeddings for UMAP
    all_emb = np.vstack([background_emb, embeddings])
    n_points = all_emb.shape[0]
    nn = max(2, min(30, n_points - 1))
    
    # Fit UMAP on combined data
    reducer = umap.UMAP(n_neighbors=nn, min_dist=0.1, metric='cosine', random_state=42)
    all_coords = reducer.fit_transform(all_emb)
    
    # Split back into background and trajectory coordinates
    background_coords = all_coords[:n_background]
    umap_coords = all_coords[n_background:]
    
    # Reshape coordinates back to trajectory format
    coords_reshaped = umap_coords.reshape(n_timepoints, n_seqs, 2)
    
    # Create the plot with HydrAMP styling
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Apply HydrAMP border styling (like densities_stylised.py)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#B8B8B8')
    ax.spines['bottom'].set_color('#B8B8B8')
    
    # Add grid with same thickness as borders (like densities_stylised.py)
    ax.grid(True, alpha=0.3, linewidth=1.2)
    ax.set_axisbelow(True)  # Grid behind plot elements
    
    # Plot each trajectory
    for seq_idx in range(n_seqs):
        trajectory = coords_reshaped[:, seq_idx, :]
        
        # Plot trajectory with color gradient
        points = np.array([trajectory[t:t+2] for t in range(n_timepoints - 1)])
        segments = points.reshape(-1, 2, 2)
        
        # Create gradient colors using custom pink to blue progression
        colors = []
        # Define the gradient colors in hex and convert to RGB
        hex_colors = ['#FF4D6D', '#D25E8A', '#A56FA7', '#7981C5', '#4C92E2', '#1FA3FF']
        rgb_colors = []
        for hex_color in hex_colors:
            # Convert hex to RGB (0-1 range)
            r = int(hex_color[1:3], 16) / 255.0
            g = int(hex_color[3:5], 16) / 255.0
            b = int(hex_color[5:7], 16) / 255.0
            rgb_colors.append((r, g, b))
        
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
            colors.append((r, g, b, 0.8))
        
        # Plot segments with color gradient
        for segment, color in zip(segments, colors):
            ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=1, zorder=1)
        
        # Plot start and end points
        start_x, start_y = trajectory[0]
        end_x, end_y = trajectory[-1]
        
        # Start point (pink/red circle - #FF4D6D)
        ax.scatter(start_x, start_y, color=(1.0, 0.302, 0.427), s=30, marker='o', zorder=2)
        
        # End point (blue circle - #1FA3FF)
        ax.scatter(end_x, end_y, color=(0.122, 0.639, 1.0), s=30, marker='o', zorder=2)
    
    # Customize the plot
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title('Guided Flow Trajectories in Sequence Space', 
                fontsize=14, fontweight='bold')
    
    # Simple legend for start/end points
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color=(1.0, 0.302, 0.427), linestyle='None',
               markersize=8, label='Start (t=0)'),
        Line2D([0], [0], marker='o', color=(0.122, 0.639, 1.0), linestyle='None',
               markersize=8, label='End (t=1)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=False)
    
    # Clean axis styling consistent with HydrAMP
    ax.tick_params(axis='both', which='major', labelsize=10, length=0)
    
    # Create gradient legend below the plot
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as patches
    
    # Create gradient colormap matching trajectory colors (pink to blue)
    colors_list = []
    n_gradient_points = 100
    # Use the same gradient colors as trajectories
    hex_colors = ['#FF4D6D', '#D25E8A', '#A56FA7', '#7981C5', '#4C92E2', '#1FA3FF']
    rgb_colors = []
    for hex_color in hex_colors:
        r = int(hex_color[1:3], 16) / 255.0
        g = int(hex_color[3:5], 16) / 255.0
        b = int(hex_color[5:7], 16) / 255.0
        rgb_colors.append((r, g, b))
    
    for i in range(n_gradient_points):
        progress = i / (n_gradient_points - 1)
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
        colors_list.append((r, g, b))
    
    gradient_cmap = LinearSegmentedColormap.from_list("trajectory_gradient", colors_list)
    
    # Position gradient bar below main plot
    gradient_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])  # [left, bottom, width, height]
    
    # Create gradient
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient_ax.imshow(gradient, aspect='auto', cmap=gradient_cmap)
    
    # Style the gradient legend
    gradient_ax.set_xlim(0, 255)
    gradient_ax.set_yticks([])
    gradient_ax.set_xticks([0, 127, 255])
    gradient_ax.set_xticklabels(['t = 0', 't = 0.5', 't = 1'], fontsize=10)
    gradient_ax.tick_params(length=0)
    
    # Add border to gradient legend
    for spine in gradient_ax.spines.values():
        spine.set_color('#B8B8B8')
        spine.set_linewidth(1.2)
    
    plt.subplots_adjust(bottom=0.15)  # Make room for gradient legend
    
    # Save figure
    out_path = Path(out_dir) / "guided_diffusion_trajectories.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nGuided trajectories plot saved to {out_path}")
    
    # Save and close (no display needed for HPC job)
    plt.close()
    
    # Print summary statistics
    print(f"\nTrajectory Analysis Summary:")
    print(f"Number of sequences: {n_seqs}")
    print(f"Diffusion steps: {n_timepoints - 1}")
    print(f"Total sequences embedded: {len(all_sequences)}")
    
    # Score initial and final sequences
    print("\nSequence Trajectories:")
    print("-" * 80)
    print(f"{'ID':>3} {'Start Sequence':<15} {'End Sequence':<15} {'Pot':>5} {'Hml':>5} {'Cyt':>5}")
    print("-" * 80)
    
    for seq_idx in range(n_seqs):
        start_seq = trajectory_info[seq_idx]['sequence']
        end_seq = trajectory_info[seq_idx + (n_timepoints-1)*n_seqs]['sequence']
        
        # Score final sequence
        tokens = x_intermediates[-1, seq_idx].unsqueeze(0)
        pot_score = pot_judge(tokens).item()
        hml_score = hml_judge(tokens).item()
        cyt_score = cyt_judge(tokens).item()
        
        print(f"{seq_idx:3d} {start_seq:<15} {end_seq:<15} {pot_score:5.2f} {hml_score:5.2f} {cyt_score:5.2f}")
    print("-" * 80)
    
    # Calculate average trajectory distances
    trajectory_distances = []
    for seq_idx in range(n_seqs):
        start_pos = coords_reshaped[0, seq_idx]
        end_pos = coords_reshaped[-1, seq_idx]
        distance = np.linalg.norm(end_pos - start_pos)
        trajectory_distances.append(distance)
    
    print(f"Average trajectory distance: {np.mean(trajectory_distances):.3f} ± {np.std(trajectory_distances):.3f}")
    print(f"This shows how far sequences move in chemical space during guidance.")

def main():
    """Main function to run the guided trajectories analysis."""
    # Use CUDA if available, otherwise CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Use moderate batch for rich visualization
    plot_guided_trajectories(
        device=device,
        batch_size=30,   # More sequences for richer patterns
        steps=200,       # More steps to show smooth trajectories
        length=12        # Standard AMP length
    )

# Auto-run when executed
if __name__ == "__main__":
    main()
