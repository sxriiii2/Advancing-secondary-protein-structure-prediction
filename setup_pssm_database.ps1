# PowerShell script to complete PSSM database setup
# Steps: Extract database, check BLAST+, format database

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "PSSM Database Setup - Steps 2-4" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Ensure databases folder exists and file is there
Write-Host "[STEP 1] Checking database file..." -ForegroundColor Yellow
if (-not (Test-Path "databases")) {
    New-Item -ItemType Directory -Path "databases" -Force | Out-Null
    Write-Host "  Created databases folder" -ForegroundColor Green
}

# Check if file exists in datasets or databases
$gzFile = $null
if (Test-Path "databases\uniref90.fasta.gz") {
    $gzFile = "databases\uniref90.fasta.gz"
    Write-Host "  [OK] Found uniref90.fasta.gz in databases folder" -ForegroundColor Green
} elseif (Test-Path "datasets\uniref90.fasta.gz") {
    Write-Host "  Found uniref90.fasta.gz in datasets folder, copying..." -ForegroundColor Yellow
    Copy-Item "datasets\uniref90.fasta.gz" -Destination "databases\uniref90.fasta.gz" -Force
    $gzFile = "databases\uniref90.fasta.gz"
    Write-Host "  [OK] File copied to databases folder" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] uniref90.fasta.gz not found!" -ForegroundColor Red
    Write-Host "  Please ensure the file is in either datasets/ or databases/ folder" -ForegroundColor Yellow
    exit 1
}

# Step 2: Check if already extracted
Write-Host ""
Write-Host "[STEP 2] Checking extraction status..." -ForegroundColor Yellow
if (Test-Path "databases\uniref90.fasta") {
    $size = (Get-Item "databases\uniref90.fasta").Length / 1GB
    Write-Host ("  [OK] Database already extracted ({0:N2} GB)" -f $size) -ForegroundColor Green
    $extract = $false
} else {
    Write-Host "  Database not extracted yet" -ForegroundColor Yellow
    $extract = $true
}

# Step 3: Extract if needed
if ($extract) {
    Write-Host ""
    Write-Host "[STEP 3] Extracting database..." -ForegroundColor Yellow
    Write-Host "  This will take 10-30 minutes and requires ~100GB free space" -ForegroundColor Cyan
    Write-Host "  Using Python to extract..." -ForegroundColor Yellow
    
    try {
        py extract_database.py
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [ERROR] Extraction failed" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "  [ERROR] Failed to run extraction: $_" -ForegroundColor Red
        Write-Host "  You can extract manually using 7-Zip or WinRAR:" -ForegroundColor Yellow
        Write-Host "    Right-click uniref90.fasta.gz -> Extract Here" -ForegroundColor White
        exit 1
    }
}

# Step 4: Check BLAST+ installation
Write-Host ""
Write-Host "[STEP 4] Checking BLAST+ installation..." -ForegroundColor Yellow

# Check common installation paths
$blastPaths = @(
    "C:\Program Files\NCBI\blast-*\bin\makeblastdb.exe",
    "C:\Program Files (x86)\NCBI\blast-*\bin\makeblastdb.exe",
    "$env:LOCALAPPDATA\Programs\NCBI\blast-*\bin\makeblastdb.exe"
)

$makeblastdb = $null
foreach ($path in $blastPaths) {
    $found = Get-ChildItem $path -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) {
        $makeblastdb = $found.FullName
        break
    }
}

# Also check if in PATH
if (-not $makeblastdb) {
    try {
        $result = Get-Command makeblastdb -ErrorAction Stop
        $makeblastdb = $result.Source
    } catch {
        # Not in PATH
    }
}

if ($makeblastdb) {
    Write-Host "  [OK] Found makeblastdb at: $makeblastdb" -ForegroundColor Green
    
    # Check psiblast too
    $psiblastPath = Join-Path (Split-Path $makeblastdb) "psiblast.exe"
    if (Test-Path $psiblastPath) {
        Write-Host "  [OK] Found psiblast at: $psiblastPath" -ForegroundColor Green
    } else {
        Write-Host "  [WARNING] psiblast not found at expected location" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [ERROR] BLAST+ not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Please install BLAST+:" -ForegroundColor Yellow
    Write-Host "  1. Download from: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/" -ForegroundColor White
    Write-Host "  2. Install: ncbi-blast-*-win64.exe" -ForegroundColor White
    Write-Host "  3. Add to PATH: C:\Program Files\NCBI\blast-*\bin\" -ForegroundColor White
    Write-Host "  4. Restart PowerShell and run this script again" -ForegroundColor White
    exit 1
}

# Step 5: Format database
Write-Host ""
Write-Host "[STEP 5] Formatting database for BLAST..." -ForegroundColor Yellow

if (Test-Path "databases\uniref90_db.phr") {
    Write-Host "  [OK] Database already formatted!" -ForegroundColor Green
    Write-Host "  Formatted database files found in databases\uniref90_db.*" -ForegroundColor Green
} else {
    Write-Host "  This will take 1-2 hours and requires ~100GB additional space" -ForegroundColor Cyan
    Write-Host "  Running: makeblastdb -in databases\uniref90.fasta -dbtype prot -out databases\uniref90_db" -ForegroundColor Yellow
    Write-Host ""
    
    $confirm = Read-Host "Continue with formatting? (yes/no)"
    if ($confirm -ne "yes") {
        Write-Host "Formatting cancelled. Run this script again when ready." -ForegroundColor Yellow
        exit 0
    }
    
    Write-Host ""
    Write-Host "Starting database formatting..." -ForegroundColor Green
    Write-Host "(This will take 1-2 hours - you can minimize this window)" -ForegroundColor Yellow
    Write-Host ""
    
    Push-Location "databases"
    try {
        & $makeblastdb -in "uniref90.fasta" -dbtype prot -out "uniref90_db"
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "  [OK] Database formatted successfully!" -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "  [ERROR] Database formatting failed" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host ""
        Write-Host "  [ERROR] Failed to format database: $_" -ForegroundColor Red
        exit 1
    } finally {
        Pop-Location
    }
}

# Summary
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Generate PSSM features:" -ForegroundColor White
Write-Host "   py generate_pssm_features.py --db databases/uniref90_db --sequences preprocessed data/train_sequences.pkl --output preprocessed data/train_pssm.pkl" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Train model (will auto-detect PSSM):" -ForegroundColor White
Write-Host "   py train_model.py" -ForegroundColor Cyan
Write-Host ""

