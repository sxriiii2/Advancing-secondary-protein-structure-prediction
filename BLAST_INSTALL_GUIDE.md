# BLAST+ Installation Guide for Windows

## Quick Installation Steps

### Step 1: Download BLAST+
1. Go to: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/
2. Download the Windows 64-bit installer:
   - Look for: `ncbi-blast-*-win64.exe` (e.g., `ncbi-blast-2.15.0+-win64.exe`)
   - This is typically ~100-200 MB

### Step 2: Install BLAST+
1. Run the downloaded `.exe` installer
2. Follow the installation wizard
3. **Default installation path**: `C:\Program Files\NCBI\blast-2.15.0+\bin\`
   (Version number may vary)

### Step 3: Add to PATH
1. Press `Win + X` and select "System"
2. Click "Advanced system settings"
3. Click "Environment Variables..."
4. Under "System variables", find and select "Path"
5. Click "Edit..."
6. Click "New" and add:
   ```
   C:\Program Files\NCBI\blast-2.15.0+\bin
   ```
   (Replace `2.15.0+` with your actual version number)
7. Click "OK" on all dialogs

### Step 4: Verify Installation
1. **Close and reopen PowerShell** (important!)
2. Run:
   ```powershell
   makeblastdb -version
   psiblast -version
   ```
3. You should see version information

### Step 5: Continue Setup
Once BLAST+ is installed and verified, run:
```powershell
.\setup_pssm_database.ps1
```

This will format the database (takes 1-2 hours).

## Alternative: Manual PATH Addition

If you can't find the exact path, search for `makeblastdb.exe`:
1. Open File Explorer
2. Navigate to `C:\Program Files\NCBI\`
3. Find the `blast-*` folder
4. Go into `bin` folder
5. Copy the full path (e.g., `C:\Program Files\NCBI\blast-2.15.0+\bin`)
6. Add this to your PATH as described above

## Troubleshooting

**Problem**: `makeblastdb` not found after adding to PATH
- **Solution**: Close and reopen PowerShell/terminal
- **Solution**: Restart your computer if needed

**Problem**: Can't find the installation folder
- **Solution**: Search for `makeblastdb.exe` in File Explorer
- **Solution**: Check `C:\Program Files (x86)\NCBI\` if not in `C:\Program Files\`

**Problem**: Permission denied
- **Solution**: Run PowerShell as Administrator

## After Installation

Once BLAST+ is installed, the setup script will:
1. ✅ Extract database (already done - 4.74 GB)
2. ⏳ Format database with `makeblastdb` (1-2 hours)
3. ✅ Ready for PSSM generation

Then you can generate PSSM features and train your model!

