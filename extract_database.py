"""Extract UniRef90 database from gzip archive"""
import gzip
import shutil
from pathlib import Path

print("=" * 60)
print("Extracting UniRef90 Database")
print("=" * 60)

# Paths
gz_file = Path("databases/uniref90.fasta.gz")
output_file = Path("databases/uniref90.fasta")

# Check if gz file exists
if not gz_file.exists():
    print(f"[ERROR] File not found: {gz_file}")
    print("Please ensure uniref90.fasta.gz is in the databases folder")
    exit(1)

# Check if already extracted
if output_file.exists():
    print(f"[INFO] {output_file} already exists")
    response = input("Extract again? (y/n): ").strip().lower()
    if response != 'y':
        print("Skipping extraction.")
        exit(0)

print(f"\nExtracting {gz_file} to {output_file}...")
print("This may take 10-30 minutes depending on your system...")
print("")

try:
    # Get file size for progress
    gz_size = gz_file.stat().st_size / (1024**3)  # GB
    print(f"Compressed size: {gz_size:.2f} GB")
    print("Expected uncompressed size: ~100 GB")
    print("")
    
    # Extract
    with gzip.open(gz_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Verify
    if output_file.exists():
        size_gb = output_file.stat().st_size / (1024**3)
        print(f"\n[OK] Extraction complete!")
        print(f"Uncompressed size: {size_gb:.2f} GB")
        print(f"File location: {output_file.absolute()}")
    else:
        print("[ERROR] Extraction may have failed - output file not found")
        
except Exception as e:
    print(f"\n[ERROR] Extraction failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("Next step: Format the database with makeblastdb")
print("=" * 60)

