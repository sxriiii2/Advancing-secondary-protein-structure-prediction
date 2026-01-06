"""
Generate PSSM (Position-Specific Scoring Matrix) features for protein sequences
Uses PSI-BLAST against UniRef90 database
"""

import subprocess
import os
from pathlib import Path
import numpy as np
import pickle
from tqdm import tqdm
import tempfile

def parse_pssm_file(pssm_file):
    """
    Parse PSI-BLAST PSSM output file
    Returns: numpy array of shape (seq_len, 20) with PSSM scores
    """
    pssm_scores = []
    
    with open(pssm_file, 'r') as f:
        lines = f.readlines()
    
    # Find where PSSM data starts (after header)
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('Last position-specific scoring matrix'):
            start_idx = i + 3  # Skip header lines
            break
    
    # Parse PSSM scores (20 columns for 20 amino acids)
    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) >= 22:  # Position + 20 scores + other info
            try:
                # Extract 20 PSSM scores (columns 2-21)
                scores = [float(parts[i]) for i in range(2, 22)]
                pssm_scores.append(scores)
            except (ValueError, IndexError):
                continue
    
    if len(pssm_scores) == 0:
        return None
    
    return np.array(pssm_scores, dtype=np.float32)

def generate_pssm_for_sequence(sequence, db_path, temp_dir=None):
    """
    Generate PSSM for a single protein sequence
    
    Args:
        sequence: Amino acid sequence string
        db_path: Path to formatted BLAST database
        temp_dir: Temporary directory for files
    
    Returns:
        PSSM array of shape (seq_len, 20) or None if failed
    """
    if temp_dir is None:
        temp_dir = Path(tempfile.gettempdir())
    
    # Create temporary FASTA file
    temp_fasta = temp_dir / f"temp_seq_{os.getpid()}.fasta"
    temp_pssm = temp_dir / f"temp_pssm_{os.getpid()}.pssm"
    
    try:
        # Write sequence to FASTA file
        with open(temp_fasta, 'w') as f:
            f.write(f">temp_sequence\n{sequence}\n")
        
        # Run PSI-BLAST
        cmd = [
            'psiblast',
            '-query', str(temp_fasta),
            '-db', str(db_path),
            '-out_ascii_pssm', str(temp_pssm),
            '-num_iterations', '3',
            '-evalue', '0.001',
            '-num_threads', '2',  # Use 2 threads for CPU
            '-max_target_seqs', '1000'
        ]
        
        # Run PSI-BLAST (suppress output)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per sequence
        )
        
        if result.returncode != 0:
            return None
        
        # Parse PSSM file
        if temp_pssm.exists():
            pssm_array = parse_pssm_file(temp_pssm)
            return pssm_array
        else:
            return None
            
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        return None
    finally:
        # Clean up temp files
        for f in [temp_fasta, temp_pssm]:
            if f.exists():
                try:
                    f.unlink()
                except:
                    pass

def generate_pssm_batch(sequences, db_path, output_file, batch_size=100):
    """
    Generate PSSM for a batch of sequences
    
    Args:
        sequences: List of amino acid sequences (as strings)
        db_path: Path to formatted BLAST database
        output_file: Output pickle file to save PSSM features
        batch_size: Process in batches
    """
    print("=" * 60)
    print("Generating PSSM Features")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Total sequences: {len(sequences)}")
    print("")
    
    # Check if database exists
    if not Path(db_path).exists():
        print(f"[ERROR] Database not found: {db_path}")
        print("Please download and format the database first!")
        return None
    
    pssm_features = []
    failed_count = 0
    
    # Process sequences
    for i, seq in enumerate(tqdm(sequences, desc="Generating PSSM")):
        # Convert sequence from indices to string if needed
        if isinstance(seq, (list, np.ndarray)):
            # Convert indices back to amino acid string
            from preprocess_data import IDX_TO_AA
            seq_str = ''.join([IDX_TO_AA.get(int(aa), 'X') for aa in seq if int(aa) < 20])
        else:
            seq_str = seq
        
        # Generate PSSM
        pssm = generate_pssm_for_sequence(seq_str, db_path)
        
        if pssm is not None:
            pssm_features.append(pssm)
        else:
            # Use zero matrix if PSSM generation fails
            pssm_features.append(np.zeros((len(seq_str), 20), dtype=np.float32))
            failed_count += 1
        
        # Save progress every batch_size
        if (i + 1) % batch_size == 0:
            with open(output_file, 'wb') as f:
                pickle.dump(pssm_features, f)
            print(f"  Progress saved: {i+1}/{len(sequences)} sequences processed")
    
    # Final save
    with open(output_file, 'wb') as f:
        pickle.dump(pssm_features, f)
    
    print("")
    print("=" * 60)
    print(f"PSSM Generation Complete!")
    print(f"Successful: {len(sequences) - failed_count}/{len(sequences)}")
    print(f"Failed: {failed_count}/{len(sequences)}")
    print(f"Saved to: {output_file}")
    print("=" * 60)
    
    return pssm_features

def main():
    """Main function to generate PSSM for preprocessed data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate PSSM features')
    parser.add_argument('--db', type=str, default='databases/uniref90_db',
                       help='Path to BLAST database')
    parser.add_argument('--sequences', type=str, default='preprocessed data/train_sequences.pkl',
                       help='Path to sequences pickle file')
    parser.add_argument('--output', type=str, default='preprocessed data/train_pssm.pkl',
                       help='Output PSSM pickle file')
    
    args = parser.parse_args()
    
    # Load sequences
    print(f"Loading sequences from {args.sequences}...")
    with open(args.sequences, 'rb') as f:
        sequences = pickle.load(f)
    
    print(f"Loaded {len(sequences)} sequences")
    
    # Generate PSSM
    pssm_features = generate_pssm_batch(
        sequences=sequences,
        db_path=args.db,
        output_file=args.output
    )
    
    if pssm_features:
        print("\n[OK] PSSM features generated successfully!")
        print(f"Use these features in training by setting use_pssm=True")

if __name__ == "__main__":
    main()

