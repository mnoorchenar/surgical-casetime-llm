param(
  [string]$Message  = "",
  [switch]$PullOnly
)

if ($Message -eq "" -and -not $PullOnly) {
  $Message = "Update " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
}

$repoRoot = git rev-parse --show-toplevel 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "Error: Not a git repository"; exit 1 }
Set-Location $repoRoot

Write-Host "`n=== GitHub Sync : surgical-casetime-llm ===`n"

# ------ STEP 0: LFS --- auto-detect and register large file extensions --------
$lfsAvailable = $false
git lfs version 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
  $lfsAvailable = $true
  git lfs install --local 2>$null | Out-Null
}

if ($lfsAvailable) {
  Write-Host "Checking for untracked large files (> 50 MB)..."
  $threshold  = 52428800
  $lfsTracked = git lfs track 2>$null
  $newExts    = @()

  Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Length -gt $threshold -and $_.FullName -notmatch '[/\\]\.git[/\\]' } |
    ForEach-Object {
      $ext = $_.Extension.ToLower()
      if ($ext -and ($lfsTracked -notmatch [regex]::Escape("*$ext"))) {
        if ($newExts -notcontains $ext) { $newExts += $ext }
      }
    }

  if ($newExts.Count -gt 0) {
    Write-Host "  [!] Large files found with untracked extensions: $($newExts -join '  ')"
    foreach ($ext in $newExts) {
      git lfs track "*$ext" | Out-Null
      Write-Host "  [OK] LFS now tracking: *$ext"
    }
    Write-Host "[OK] .gitattributes updated"
  } else {
    Write-Host "[OK] All large files are already LFS-tracked (or none present)"
  }
} else {
  Write-Host "[WARN] Git LFS not found --- files over 100 MB may be rejected by GitHub"
  Write-Host "       Install: winget install -e --id GitHub.GitLFS"
}

# ------ STEP 1: Stage everything ----------------------------------------------
Write-Host "`nStaging changes..."
git add -A
Write-Host "[OK] Staged"

# ------ STEP 2: Fetch + merge from GitHub -------------------------------------
Write-Host "`nFetching from GitHub..."
git fetch origin 2>$null
$remoteExists = $LASTEXITCODE -eq 0

if ($remoteExists) {
  $localCommit  = git rev-parse HEAD 2>$null
  $remoteCommit = git rev-parse origin/main 2>$null
  if ($localCommit -ne $remoteCommit) {
    Write-Host "Merging from GitHub..."
    git diff HEAD --quiet
    if ($LASTEXITCODE -ne 0) { git commit -m "Local changes before merge" 2>$null | Out-Null }
    git merge origin/main --no-edit 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
      Write-Host "[!] Merge conflict --- keeping local version..."
      git merge --abort 2>$null
    } else {
      Write-Host "[OK] Merged from GitHub"
    }
  } else {
    Write-Host "[OK] Already up-to-date"
  }
}

if ($PullOnly) { Write-Host "`n[OK] Pull complete`n"; exit 0 }

# ------ STEP 3: Commit --------------------------------------------------------
git diff HEAD --quiet
if ($LASTEXITCODE -ne 0) {
  git commit -m $Message
  Write-Host "[OK] Committed: $Message"
} else {
  Write-Host "[OK] Nothing new to commit"
}

# ------ STEP 4: Push ----------------------------------------------------------
Write-Host "`nPushing to GitHub..."
git push origin main 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
  Write-Host "[OK] Push complete"
} else {
  Write-Host "[WARN] Normal push failed --- trying with LFS retry..."
  git lfs push origin main --all 2>&1 | Out-Null
  git push origin main 2>&1 | Out-Null
  if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Push complete (after LFS retry)"
  } else {
    git push origin main --force 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { Write-Host "[OK] Push complete (forced)" }
    else                      { Write-Host "[ERROR] Push failed"; exit 1 }
  }
}

Write-Host "`n[OK] Sync complete`n"
