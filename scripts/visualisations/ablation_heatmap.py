#!/usr/bin/env python3
"""ablation_heatmap.py

Create HydrAMP-style heatmap showing amino acid correlations with peptide properties
from the PepDFM ablation analysis results.

Designed to be run directly in Jupyter notebook cells without saving plots.

Usage in Jupyter cell:
    %run ablation_heatmap.py
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

# Default paths (following the pattern from other scripts)
DEFAULT_RESULTS_DIR = Path.home() / "mog_dfm/ampflow/results/mog/ablations_analysis"
DEFAULT_AA_ASSOCIATIONS_CSV = DEFAULT_RESULTS_DIR / "aa_associations_toxicity.csv"

# Additional generic target data paths
DEFAULT_GENERIC_HP_DIR = Path.home() / "mog_dfm/ampflow/results/mog/generic_hp4_111_2500"
DEFAULT_GENERIC_DIR = Path.home() / "mog_dfm/ampflow/results/mog/generic"

# Ablation CSV glob pattern
# -----------------------------------------------------------------------------
ABLATIONS_ROOT = Path.home() / "mog_dfm/ampflow/results/mog/ablations"


def has_long_homopolymer(seq: str, max_run: int = 3) -> bool:
    """Return True if *any* amino-acid occurs in a run > max_run."""
    if not seq:
        return False
    current = seq[0]
    count = 1
    for aa in seq[1:]:
        if aa == current:
            count += 1
            if count > max_run:
                return True
        else:
            current = aa
            count = 1
    return False


def load_and_filter_sequences(csv_path: Path, max_length: int = 30) -> pd.DataFrame:
    """Load sequences from CSV and apply length and homopolymer filtering.
    
    Parameters
    ----------
    csv_path : Path
        Path to mog_samples_scores.csv file
    max_length : int
        Maximum sequence length to keep (default 30)
        
    Returns
    -------
    pd.DataFrame
        Filtered dataframe with sequences <max_length and no long homopolymers
    """
    if not csv_path.exists():
        return pd.DataFrame()
        
    df = pd.read_csv(csv_path)
    
    # Apply filters
    df = df[df["sequence"].str.len() < max_length]
    df = df[~df["sequence"].apply(has_long_homopolymer)]
    
    return df


def load_all_ablation_data() -> pd.DataFrame:
    """Load all ablation sequences from CSV files and compute filtered dataset.
    
    Returns
    -------
    pd.DataFrame
        Combined and filtered sequences from all ablation runs plus generic folders
    """
    frames = []
    
    # Load all ablation CSVs
    if ABLATIONS_ROOT.exists():
        for csv_path in ABLATIONS_ROOT.glob("**/mog_samples_scores.csv"):
            df = load_and_filter_sequences(csv_path)
            if not df.empty:
                df["source"] = csv_path.parent.name
                frames.append(df)
    
    # Load baseline generic data
    generic_csv = DEFAULT_GENERIC_DIR / "mog_samples_scores.csv"
    if generic_csv.exists():
        df_base = load_and_filter_sequences(generic_csv)
        if not df_base.empty:
            df_base["source"] = "generic_baseline"
            frames.append(df_base)
    
    # Load homopolymer-penalised generic data  
    generic_hp_csv = DEFAULT_GENERIC_HP_DIR / "mog_samples_scores.csv"
    if generic_hp_csv.exists():
        df_hp = load_and_filter_sequences(generic_hp_csv)
        if not df_hp.empty:
            df_hp["source"] = "generic_hp4_111_2500"
            frames.append(df_hp)
    
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        print(f"Loaded {len(combined)} filtered sequences from {len(frames)} sources")
        return combined
    else:
        print("No sequence data found")
        return pd.DataFrame()


def compute_aa_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute amino acid correlations with peptide properties from raw sequence data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: sequence, potency, hemolysis, cytotox
        
    Returns
    -------
    pd.DataFrame
        Correlation matrix with amino acids as rows and properties as columns
    """
    if df.empty:
        return pd.DataFrame()
    
    # Standard amino acids
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # One-hot encode amino acids
    aa_matrix = np.zeros((len(df), len(amino_acids)))
    for i, seq in enumerate(df["sequence"]):
        for aa in seq:
            if aa in amino_acids:
                aa_idx = amino_acids.index(aa)
                aa_matrix[i, aa_idx] += 1
    
    # Normalize by sequence length to get frequencies
    seq_lengths = df["sequence"].str.len().values.reshape(-1, 1)
    aa_freq = aa_matrix / seq_lengths
    
    # Prepare property scores (toxicity = 1 - safety)
    properties = {
        'potency': df["potency"].values,
        'hemolysis': 1 - df["hemolysis"].values,  # Convert safety to toxicity
        'cytotox': 1 - df["cytotox"].values      # Convert safety to toxicity
    }
    
    # Compute Spearman correlations
    correlations = {}
    for prop_name, prop_values in properties.items():
        prop_corrs = []
        for aa_idx in range(len(amino_acids)):
            corr, _ = spearmanr(aa_freq[:, aa_idx], prop_values)
            prop_corrs.append(corr if not np.isnan(corr) else 0.0)
        correlations[prop_name] = prop_corrs
    
    # Create correlation matrix
    correlation_matrix = pd.DataFrame(correlations, index=amino_acids)
    
    # Reorder rows to match standard AA grouping
    aa_order = ['A', 'V', 'I', 'L', 'M',
                'F', 'W', 'Y', 
                'S', 'T', 'N', 'Q', 'C',
                'K', 'R', 'H',
                'D', 'E',
                'G', 'P']
    
    correlation_matrix = correlation_matrix.reindex(aa_order)
    
    print(f"Computed correlations for {len(correlation_matrix)} amino acids and {len(correlation_matrix.columns)} properties")
    return correlation_matrix





def plot_radial_aa_heatmap(correlation_matrix: pd.DataFrame):
    """Create radial heatmap with amino acids around circumference and properties as concentric rings.
    
    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        Matrix with amino acids as rows and properties as columns
    """
    from matplotlib.colors import Normalize
    from matplotlib.cm import RdBu_r
    
    # Create figure with standard axes (not polar)
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Set up colormap
    norm = Normalize(vmin=-0.7, vmax=0.7)
    cmap = RdBu_r
    
    # Group amino acids by chemical properties for better visual organization
    aa_groups = {
        'Hydrophobic': ['A', 'V', 'I', 'L', 'M'],
        'Aromatic': ['F', 'W', 'Y'],
        'Uncharged polar': ['S', 'T', 'N', 'Q', 'C'],
        'Cationic': ['K', 'R', 'H'],
        'Anionic': ['D', 'E'],
        'Structural': ['G', 'P']
    }
    
    # Create ordered list maintaining chemical grouping
    aa_ordered = []
    for group in ['Hydrophobic', 'Aromatic', 'Uncharged polar', 'Cationic', 'Anionic', 'Structural']:
        for aa in aa_groups[group]:
            if aa in correlation_matrix.index:
                aa_ordered.append(aa)
    
    # Add any remaining amino acids
    for aa in correlation_matrix.index:
        if aa not in aa_ordered:
            aa_ordered.append(aa)
    
    # Calculate positions
    n_aa = len(aa_ordered)
    angles = np.linspace(0, 2*np.pi, n_aa, endpoint=False)
    angles_dict = {aa: angle for aa, angle in zip(aa_ordered, angles)}
    
    # Property labels and ring positions
    property_labels = {
        'potency': 'Antimicrobial Activity',
        'hemolysis': 'Hemolytic Toxicity',
        'cytotox': 'Cytotoxicity'
    }
    
    # Concentric ring order: outer=Antimicrobial, middle=Hemolysis, inner=Cytotoxicity
    # We map indices accordingly when iterating columns
    desired_order = ['potency', 'hemolysis', 'cytotox']
    available_cols = [c for c in desired_order if c in correlation_matrix.columns]
    
    ring_radii = [5.0, 3.5, 2.0]  # Outer to inner rings
    ring_width = 1.2
    
    # Plot each property as a concentric ring
    for ring_idx, prop in enumerate(available_cols):
        if prop not in correlation_matrix.columns:
            continue
            
        radius = ring_radii[ring_idx]
        
        # Create segments for each amino acid
        segment_angle = 2 * np.pi / n_aa
        
        for i, aa in enumerate(aa_ordered):
            if aa in correlation_matrix.index:
                angle = angles_dict[aa]
                value = correlation_matrix.loc[aa, prop]
                
                if pd.notna(value):
                    # Color based on correlation
                    color = cmap(norm(value))
                    
                    # Create segment as a thick arc
                    start_angle = angle - segment_angle/2
                    end_angle = angle + segment_angle/2
                    
                    # Draw filled arc segment
                    theta = np.linspace(start_angle, end_angle, 50)
                    x_inner = (radius - ring_width/2) * np.cos(theta)
                    y_inner = (radius - ring_width/2) * np.sin(theta)
                    x_outer = (radius + ring_width/2) * np.cos(theta)
                    y_outer = (radius + ring_width/2) * np.sin(theta)
                    
                    # Create polygon for the segment
                    x_coords = np.concatenate([x_inner, x_outer[::-1]])
                    y_coords = np.concatenate([y_inner, y_outer[::-1]])
                    
                    ax.fill(x_coords, y_coords, color=color, alpha=0.8, 
                           edgecolor='white', linewidth=0.5)
                    
                    # Add value annotation for strong correlations
                    if abs(value) > 0.3:
                        text_x = radius * np.cos(angle)
                        text_y = radius * np.sin(angle)
                        ax.text(text_x, text_y, f'{value:.2f}', 
                               ha='center', va='center', fontsize=8,
                               color='white' if abs(value) > 0.5 else 'black',
                               fontweight='bold' if abs(value) > 0.5 else 'normal')
    
    # Add amino acid labels around the outside
    label_radius = max(ring_radii) + ring_width + 0.8
    for aa in aa_ordered:
        angle = angles_dict[aa]
        x = label_radius * np.cos(angle)
        y = label_radius * np.sin(angle)
        ax.text(x, y, aa, ha='center', va='center', 
               fontsize=12, fontweight='bold')
    
    # Add property ring labels positioned to avoid overlap
    label_positions = [
        (ring_radii[0], np.pi/4),  # Top right (Antimicrobial - outer)
        (ring_radii[1], 3*np.pi/4),  # Top left  (Hemolysis - middle)
        (ring_radii[2], np.pi/8)   # Between V and A segments (Cytotoxicity - inner)
    ]
    
    for ring_idx, prop in enumerate(available_cols):
        if ring_idx < len(label_positions):
            radius, label_angle = label_positions[ring_idx]
            x = radius * np.cos(label_angle)
            y = radius * np.sin(label_angle)
            label = property_labels.get(prop, prop)
            ax.text(x, y, label, ha='center', va='center', 
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Add chemical group separators and labels
    current_idx = 0
    group_boundaries = []
    
    for i, group in enumerate(['Hydrophobic', 'Aromatic', 'Uncharged polar', 'Cationic', 'Anionic', 'Structural']):
        group_aas = [aa for aa in aa_groups[group] if aa in aa_ordered]
        if group_aas:
            start_idx = current_idx
            end_idx = current_idx + len(group_aas)
            
            # Store boundary for separator line after each group except the last
            if i < 5:  # boundaries after all groups except the last
                boundary_angle = angles[end_idx] - segment_angle/2
                group_boundaries.append(boundary_angle)
    
            # Group label at center of group with custom positioning
            mid_idx = start_idx + len(group_aas)//2
            mid_angle = angles[mid_idx]
            
            # Custom positioning for specific groups
            if group == 'Uncharged polar':
                group_label_radius = label_radius + 1.0  # Closer to plot
            elif group == 'Aromatic':
                group_label_radius = label_radius + 1.0  # Slightly lower
                mid_angle = mid_angle + 0.0  # Adjust angle slightly down
            elif group == 'Cationic':
                group_label_radius = label_radius + 1.0  # Closer to plot
            elif group == 'Structural':
                group_label_radius = label_radius + 1.0  # Closer to plot
                mid_angle = mid_angle - 0.2  # Move left
            elif group == 'Hydrophobic':
                group_label_radius = label_radius + 1.0  # Push in slightly
            elif group == 'Anionic':
                group_label_radius = label_radius + 1.0  # Push in slightly
            else:
                group_label_radius = label_radius + 1.5
            
            x = group_label_radius * np.cos(mid_angle)
            y = group_label_radius * np.sin(mid_angle)
            
            # Adjust positioning for better readability
            ha = 'center'
            va = 'center'
            if mid_angle > np.pi/4 and mid_angle < 3*np.pi/4:  # Top half
                va = 'bottom'
            elif mid_angle > 5*np.pi/4 and mid_angle < 7*np.pi/4:  # Bottom half
                va = 'top'
            if mid_angle > 3*np.pi/4 and mid_angle < 5*np.pi/4:  # Left side
                ha = 'right'
            elif mid_angle < np.pi/4 or mid_angle > 7*np.pi/4:  # Right side
                ha = 'left'
                
            ax.text(x, y, group, ha=ha, va=va,
                   fontsize=10, style='italic', color='gray', fontweight='bold')
            
            current_idx += len(group_aas)
    
    # Add the wrap-around boundary from end of last group back to start of first
    wrap_boundary = angles[0] - segment_angle/2  # Just before the first amino acid
    group_boundaries.append(wrap_boundary)
    
    # Draw separator lines between groups (light gray dashed)
    for i, boundary_angle in enumerate(group_boundaries):
        x1, y1 = 1.2 * np.cos(boundary_angle), 1.2 * np.sin(boundary_angle)
        x2, y2 = (label_radius + 0.3) * np.cos(boundary_angle), (label_radius + 0.3) * np.sin(boundary_angle)
        ax.plot([x1, x2], [y1, y2], color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Set equal aspect ratio and limits
    max_radius = label_radius + 2
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_aspect('equal')
    ax.axis('off')  # Remove axes
    
    # Add title
    ax.set_title('Amino Acid Correlations with Peptide Properties', 
                fontsize=16, fontweight='normal', pad=5)
    
    # Add colorbar closer to plot
    plt.subplots_adjust(bottom=0.12)
    cbar_ax = fig.add_axes([0.25, 0.105, 0.5, 0.03])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                     cax=cbar_ax, orientation='horizontal')
    cb.set_label('Spearman correlation', fontsize=12)
    cb.ax.tick_params(labelsize=10)
    
    # Save plot
    output_path = Path.home() / "mog_dfm/ampflow/plots/property_aa_associations.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Always display in Jupyter
    from IPython.display import display
    display(plt.gcf())
    plt.close()

def print_top_associations_table(correlation_matrix: pd.DataFrame, n_top: int = 5):
    """Create a summary table showing top positive and negative associations for each property.
    
    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        Matrix with amino acids as rows and properties as columns
    n_top : int, optional
        Number of top associations to show for each property (default: 5)
    """
    # Create figure for the table with better proportions
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # Property labels for display (note: hemolysis and cytotox now show toxicity, not safety)
    property_labels = {
        'potency': 'Antimicrobial Activity',
        'hemolysis': 'Hemolytic Toxicity', 
        'cytotox': 'Cytotoxicity'
    }
    
    # Build summary table data
    table_data = []
    
    for prop in correlation_matrix.columns:
        prop_data = correlation_matrix[prop].dropna().sort_values(ascending=False)
        
        # Top positive correlations
        top_positive = prop_data.head(n_top)
        # Top negative correlations  
        top_negative = prop_data.tail(n_top)[::-1]  # Reverse to show most negative first
        
        # Format as strings with correlation values
        pos_strings = [f"{aa} ({corr:.3f})" for aa, corr in top_positive.items()]
        neg_strings = [f"{aa} ({corr:.3f})" for aa, corr in top_negative.items()]
        
        table_data.append([
            property_labels.get(prop, prop),
            '\n'.join(pos_strings),
            '\n'.join(neg_strings)
        ])
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Property', f'Top {n_top} Positive', f'Top {n_top} Negative'],
        cellLoc='left',
        loc='center',
        colWidths=[0.25, 0.375, 0.375]
    )
    
    # Style the table to match HydrAMP theme
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)  # Make cells taller
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#B8B8B8')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.15)
    
    # Style data cells
    for i in range(1, len(table_data) + 1):
        for j in range(3):
            table[(i, j)].set_facecolor('white')
            table[(i, j)].set_edgecolor('#B8B8B8')
            table[(i, j)].set_linewidth(1.2)
    
    ax.set_title('Top Amino Acid Associations by Property', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Always display in Jupyter
    from IPython.display import display
    display(plt.gcf())
    plt.close()

def plot_ablation_heatmaps():
    """Generate radial amino acid correlation heatmap from raw ablation data."""
    print("Loading all ablation sequence data...")
    
    # Load all sequence data
    df = load_all_ablation_data()
    if df.empty:
        print("No data found!")
        return
    
    # Compute correlations from raw data
    print("Computing amino acid correlations...")
    correlation_matrix = compute_aa_correlations(df)
    
    print("Generating radial correlation heatmap...")
    plot_radial_aa_heatmap(correlation_matrix)
    
    print("Generating top associations summary...")
    # Simple print version
    print("\n" + "="*80)
    print("TOP AMINO ACID ASSOCIATIONS BY PROPERTY")
    print("="*80)
    
    property_labels = {
        'potency': 'Antimicrobial Activity',
        'hemolysis': 'Hemolytic Toxicity', 
        'cytotox': 'Cytotoxicity'
    }
    
    for prop in correlation_matrix.columns:
        prop_name = property_labels.get(prop, prop)
        print(f"\n{prop_name.upper()}:")
        print("-" * len(prop_name))
        
        sorted_corrs = correlation_matrix[prop].sort_values(ascending=False)
        
        # Top 5 positive associations
        print(f"\nTop 5 POSITIVE correlations (higher values):")
        top_pos = sorted_corrs.head(5)
        for i, (aa, corr) in enumerate(top_pos.items(), 1):
            print(f"  {i}. {aa}: {corr:.3f}")
        
        # Top 5 negative associations  
        print(f"\nTop 5 NEGATIVE correlations (lower values):")
        top_neg = sorted_corrs.tail(5)
        for i, (aa, corr) in enumerate(reversed(list(top_neg.items())), 1):
            print(f"  {i}. {aa}: {corr:.3f}")
        
        print()
    
    print("="*80)
    print("Ablation heatmap analysis completed!")

# Auto-run when imported or executed
plot_ablation_heatmaps()
