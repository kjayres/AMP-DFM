#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    amp_dfm_root = Path("amp_dfm")
    input_dir = amp_dfm_root / "data" / "filtered"
    output_dir = amp_dfm_root / "data" / "clustered"
    emb_dir = amp_dfm_root / "data" / "embeddings"
    
    print("Loading clustering results...")
    clu = pd.read_csv(emb_dir / "clusters_cluster.tsv", sep='\t', names=["member", "representative"])
    
    print("Creating complete sequence-to-cluster mapping...")
    all_members = set(clu['member'])
    all_representatives = set(clu['representative'])
    all_sequences = all_members | all_representatives
    
    singletons = all_representatives - all_members
    
    print(f"Total sequences in clustering: {len(all_sequences)}")
    print(f"Clustered sequences (members): {len(all_members)}")
    print(f"Singleton sequences: {len(singletons)}")
    
    seq_to_cluster = {}
    
    for _, row in clu.iterrows():
        seq_to_cluster[row['member']] = row['representative']
    
    for singleton in singletons:
        seq_to_cluster[singleton] = singleton
    
    print(f"Complete mapping size: {len(seq_to_cluster)}")
    
    print("Loading sequence mapping...")
    seqs = [line.strip() for line in open(emb_dir / "seqs.txt")]
    id_to_seq = {f"seq{i+1}": seq for i, seq in enumerate(seqs)}
    
    expected_ids = set(id_to_seq.keys())
    mapped_ids = set(seq_to_cluster.keys())
    missing_ids = expected_ids - mapped_ids
    
    if missing_ids:
        print(f"WARNING: {len(missing_ids)} sequences missing from cluster mapping!")
        print(f"First 10 missing: {list(missing_ids)[:10]}")
    else:
        print("✓ All sequences successfully mapped to clusters")
    
    print("Assigning train/val/test splits...")
    unique_clusters = list(set(seq_to_cluster.values()))
    np.random.seed(42)
    np.random.shuffle(unique_clusters)
    
    n_clusters = len(unique_clusters)
    train_end = int(0.8 * n_clusters)
    val_end = int(0.9 * n_clusters)
    
    cluster_to_split = {}
    for i, cluster in enumerate(unique_clusters):
        if i < train_end:
            cluster_to_split[cluster] = 'train'
        elif i < val_end:
            cluster_to_split[cluster] = 'val'
        else:
            cluster_to_split[cluster] = 'test'
    
    print(f"Cluster splits: {len([c for c in cluster_to_split.values() if c == 'train'])} train, "
          f"{len([c for c in cluster_to_split.values() if c == 'val'])} val, "
          f"{len([c for c in cluster_to_split.values() if c == 'test'])} test")
    
    datasets = [
        ("activities_final_long.csv", "activities_with_splits.csv"),
        ("haemolysis_final_long.csv", "haemolysis_with_splits.csv"),
        ("cytotoxicity_final_long.csv", "cytotoxicity_with_splits.csv"),
        ("negatives_swissprot_long.csv", "negatives_swissprot_with_splits.csv"),
        ("negatives_synthetic_long_filtered.csv", "negatives_synth_with_splits.csv"),
        ("negatives_general_peptides_long.csv", "negatives_general_peptides_with_splits.csv"),
        ("negatives_uniprot_long.csv", "negatives_uniprot_with_splits.csv"),
    ]
    
    for input_file, output_file in datasets:
        print(f"\nProcessing {input_file}...")
        
        df = pd.read_csv(input_dir / input_file)
        print(f"Loaded {len(df)} rows")
        
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"After removing duplicates: {len(df)} rows")
        
        seq_to_id = {seq: seq_id for seq_id, seq in id_to_seq.items()}
        df['sequence_id'] = df['sequence'].map(seq_to_id)
        
        unmapped = df[df['sequence_id'].isna()]
        if len(unmapped) > 0:
            print(f"WARNING: {len(unmapped)} sequences not found in embedding cache")
            unmapped_seqs = unmapped['sequence'].unique()
            with open(emb_dir / f"unmapped_{input_file.split('_')[0]}.txt", 'w') as f:
                f.write('\n'.join(unmapped_seqs))
            df = df[df['sequence_id'].notna()].reset_index(drop=True)
        
        df['cluster_id'] = df['sequence_id'].map(seq_to_cluster)
        
        unmapped_clusters = df[df['cluster_id'].isna()]
        if len(unmapped_clusters) > 0:
            print(f"ERROR: {len(unmapped_clusters)} sequences couldn't be mapped to clusters!")
        else:
            print("✓ All sequences successfully mapped to clusters")
        
        df['split'] = df['cluster_id'].map(cluster_to_split)
        
        output_path = output_dir / output_file
        df.to_csv(output_path, index=False)
        
        split_counts = df['split'].value_counts()
        print(f"Split distribution: {dict(split_counts)}")
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()