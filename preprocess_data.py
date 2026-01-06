"""
Data Preprocessing for Protein Secondary Structure Prediction
Handles multiple dataset formats and prepares data for 2D CNN + RNN + BiGRU model
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

# Amino acid mapping
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {idx: aa for aa, idx in AA_TO_IDX.items()}
VOCAB_SIZE = len(AMINO_ACIDS)

# Secondary structure mapping (3-state)
SS3_MAP = {'H': 0, 'E': 1, 'C': 2}  # Helix, Sheet, Coil
SS3_REVERSE = {0: 'H', 1: 'E', 2: 'C'}

class ProteinDataset(Dataset):
    """PyTorch Dataset for protein sequences and secondary structures"""
    def __init__(self, sequences, structures, max_len=700, pad_token=0, pssm_features=None):
        self.sequences = sequences
        self.structures = structures
        self.max_len = max_len
        self.pad_token = pad_token
        self.pssm_features = pssm_features  # Optional PSSM features
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        struct = self.structures[idx]
        original_len = len(seq)
        
        # Pad or truncate to max_len
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
            struct = struct[:self.max_len]
            if self.pssm_features is not None:
                pssm = self.pssm_features[idx][:self.max_len]
        else:
            pad_len = self.max_len - len(seq)
            seq = np.pad(seq, (0, pad_len), constant_values=self.pad_token)
            struct = np.pad(struct, (0, pad_len), constant_values=self.pad_token)
            if self.pssm_features is not None:
                pssm = np.pad(self.pssm_features[idx], ((0, pad_len), (0, 0)), constant_values=0.0)
        
        result = [torch.LongTensor(seq), torch.LongTensor(struct), original_len]
        
        # Add PSSM if available
        if self.pssm_features is not None:
            result.append(torch.FloatTensor(pssm))
        
        return tuple(result)

def encode_sequence(sequence):
    """Encode amino acid sequence to indices"""
    return [AA_TO_IDX.get(aa, 0) for aa in sequence.upper() if aa in AMINO_ACIDS]

def encode_structure(structure):
    """Encode secondary structure to indices (3-state)"""
    return [SS3_MAP.get(ss, 2) for ss in structure.upper()]

def load_cb513_from_files(data_dir):
    """Load CB513 dataset from .all files"""
    sequences = []
    structures = []
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Look for .all files or other formats
    files = list(data_path.glob("*.all")) + list(data_path.glob("*.fasta")) + list(data_path.glob("*.txt"))
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Parse CB513 format (sequence and structure on separate lines)
            lines = content.strip().split('\n')
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    seq_line = lines[i].strip()
                    struct_line = lines[i + 1].strip()
                    
                    if seq_line and struct_line and len(seq_line) == len(struct_line):
                        seq_encoded = encode_sequence(seq_line)
                        struct_encoded = encode_structure(struct_line)
                        
                        if len(seq_encoded) == len(struct_encoded):
                            sequences.append(seq_encoded)
                            structures.append(struct_encoded)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return sequences, structures

def load_from_huggingface(data_dir):
    """Load dataset from HuggingFace format"""
    sequences = []
    structures = []
    
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))
    
    for json_file in json_files:
        try:
            # Try JSONL format (one JSON object per line) - most common for HuggingFace
            with open(json_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Parse as JSONL
            for line in lines:
                try:
                    item = json.loads(line.strip())
                    # Handle different field names
                    seq_str = None
                    struct_str = None
                    
                    # Try different field name variations
                    if 'input' in item:
                        seq_str = item['input']
                    elif 'sequence' in item:
                        seq_str = item['sequence']
                    elif 'seq' in item:
                        seq_str = item['seq']
                    
                    if 'dssp3' in item:
                        struct_str = item['dssp3']
                    elif 'structure' in item:
                        struct_str = item['structure']
                    elif 'ss' in item:
                        struct_str = item['ss']
                    elif 'secondary_structure' in item:
                        struct_str = item['secondary_structure']
                    
                    if seq_str and struct_str:
                        seq = encode_sequence(seq_str)
                        struct = encode_structure(struct_str)
                        if len(seq) == len(struct) and len(seq) > 0:
                            sequences.append(seq)
                            structures.append(struct)
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    continue
            
            # If no data loaded, try regular JSON
            if len(sequences) == 0:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different HuggingFace dataset formats
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            seq_str = item.get('input') or item.get('sequence') or item.get('seq')
                            struct_str = item.get('dssp3') or item.get('structure') or item.get('ss')
                            if seq_str and struct_str:
                                seq = encode_sequence(seq_str)
                                struct = encode_structure(struct_str)
                                if len(seq) == len(struct) and len(seq) > 0:
                                    sequences.append(seq)
                                    structures.append(struct)
                elif isinstance(data, dict):
                    # Handle dict format - might have 'data' key or similar
                    if 'data' in data:
                        for item in data['data']:
                            seq_str = item.get('input') or item.get('sequence') or item.get('seq')
                            struct_str = item.get('dssp3') or item.get('structure') or item.get('ss')
                            if seq_str and struct_str:
                                seq = encode_sequence(seq_str)
                                struct = encode_structure(struct_str)
                                if len(seq) == len(struct) and len(seq) > 0:
                                    sequences.append(seq)
                                    structures.append(struct)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    return sequences, structures

def load_from_csv(csv_path):
    """Load dataset from CSV file"""
    df = pd.read_csv(csv_path)
    sequences = []
    structures = []
    
    # Try to find sequence and structure columns
    seq_col = None
    struct_col = None
    
    for col in df.columns:
        if 'sequence' in col.lower() or 'seq' in col.lower():
            seq_col = col
        if 'structure' in col.lower() or 'ss' in col.lower() or 'secondary' in col.lower():
            struct_col = col
    
    if seq_col and struct_col:
        for _, row in df.iterrows():
            seq = encode_sequence(str(row[seq_col]))
            struct = encode_structure(str(row[struct_col]))
            if len(seq) == len(struct):
                sequences.append(seq)
                structures.append(struct)
    
    return sequences, structures

def preprocess_all_datasets():
    """Preprocess all available datasets"""
    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    all_sequences = []
    all_structures = []
    
    # 1. Try Kaggle datasets
    kaggle_dir = Path("datasets/kaggle")
    if kaggle_dir.exists():
        print("Processing Kaggle datasets...")
        try:
            # Look for CSV files
            csv_files = list(kaggle_dir.rglob("*.csv"))
            for csv_file in csv_files:
                seqs, structs = load_from_csv(csv_file)
                all_sequences.extend(seqs)
                all_structures.extend(structs)
                print(f"  Loaded {len(seqs)} sequences from {csv_file.name}")
        except Exception as e:
            print(f"  Error processing Kaggle data: {e}")
    
    # 2. Try HuggingFace datasets
    hf_dir = Path("datasets/huggingface")
    if hf_dir.exists():
        print("Processing HuggingFace datasets...")
        for dataset_dir in hf_dir.iterdir():
            if dataset_dir.is_dir():
                try:
                    seqs, structs = load_from_huggingface(dataset_dir)
                    all_sequences.extend(seqs)
                    all_structures.extend(structs)
                    print(f"  Loaded {len(seqs)} sequences from {dataset_dir.name}")
                except Exception as e:
                    print(f"  Error processing {dataset_dir.name}: {e}")
    
    # 3. Try direct downloads
    direct_dir = Path("datasets/direct")
    if direct_dir.exists():
        print("Processing direct downloads...")
        try:
            # Extract if needed
            import tarfile
            for tgz_file in direct_dir.glob("*.tgz"):
                extract_dir = direct_dir / tgz_file.stem
                extract_dir.mkdir(exist_ok=True)
                with tarfile.open(tgz_file, 'r:gz') as tar:
                    tar.extractall(extract_dir)
                
                seqs, structs = load_cb513_from_files(extract_dir)
                all_sequences.extend(seqs)
                all_structures.extend(structs)
                print(f"  Loaded {len(seqs)} sequences from {tgz_file.name}")
        except Exception as e:
            print(f"  Error processing direct downloads: {e}")
    
    print(f"\nTotal sequences loaded: {len(all_sequences)}")
    
    if len(all_sequences) == 0:
        print("[WARNING] No sequences found! Please download datasets first.")
        return None, None
    
    # Filter sequences
    print("\nFiltering sequences...")
    filtered_seqs = []
    filtered_structs = []
    
    for seq, struct in zip(all_sequences, all_structures):
        if len(seq) > 0 and len(seq) == len(struct):
            filtered_seqs.append(seq)
            filtered_structs.append(struct)
    
    print(f"Valid sequences: {len(filtered_seqs)}")
    
    # Statistics
    seq_lens = [len(seq) for seq in filtered_seqs]
    print(f"\nSequence length statistics:")
    print(f"  Min: {min(seq_lens)}")
    print(f"  Max: {max(seq_lens)}")
    print(f"  Mean: {np.mean(seq_lens):.2f}")
    print(f"  Median: {np.median(seq_lens):.2f}")
    
    # Structure distribution
    all_structs_flat = [ss for struct in filtered_structs for ss in struct]
    struct_counts = Counter(all_structs_flat)
    print(f"\nStructure distribution:")
    for ss_id, count in sorted(struct_counts.items()):
        print(f"  {SS3_REVERSE[ss_id]}: {count} ({count/len(all_structs_flat)*100:.2f}%)")
    
    return filtered_seqs, filtered_structs

def create_splits(sequences, structures, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Create train/validation/test splits"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    n = len(sequences)
    indices = np.random.permutation(n)
    
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_seqs = [sequences[i] for i in train_indices]
    train_structs = [structures[i] for i in train_indices]
    
    val_seqs = [sequences[i] for i in val_indices]
    val_structs = [structures[i] for i in val_indices]
    
    test_seqs = [sequences[i] for i in test_indices]
    test_structs = [structures[i] for i in test_indices]
    
    return (train_seqs, train_structs), (val_seqs, val_structs), (test_seqs, test_structs)

def save_processed_data(sequences, structures, output_dir="preprocessed data"):
    """Save processed data"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save as pickle
    with open(output_path / "sequences.pkl", "wb") as f:
        pickle.dump(sequences, f)
    
    with open(output_path / "structures.pkl", "wb") as f:
        pickle.dump(structures, f)
    
    print(f"\n[OK] Processed data saved to {output_path}/")

def main():
    """Main preprocessing function"""
    # Preprocess all datasets
    sequences, structures = preprocess_all_datasets()
    
    if sequences is None:
        return
    
    # Create splits
    print("\n" + "=" * 60)
    print("Creating train/val/test splits...")
    (train_seqs, train_structs), (val_seqs, val_structs), (test_seqs, test_structs) = \
        create_splits(sequences, structures)
    
    print(f"Train: {len(train_seqs)} sequences")
    print(f"Validation: {len(val_seqs)} sequences")
    print(f"Test: {len(test_seqs)} sequences")
    
    # Save processed data
    save_processed_data(sequences, structures)
    
    # Save splits separately
    splits = {
        'train': (train_seqs, train_structs),
        'val': (val_seqs, val_structs),
        'test': (test_seqs, test_structs)
    }
    
    for split_name, (seqs, structs) in splits.items():
        with open(f"preprocessed data/{split_name}_sequences.pkl", "wb") as f:
            pickle.dump(seqs, f)
        with open(f"preprocessed data/{split_name}_structures.pkl", "wb") as f:
            pickle.dump(structs, f)
    
    print("\n[OK] All data preprocessing complete!")
    print("Ready for training!")

if __name__ == "__main__":
    main()

