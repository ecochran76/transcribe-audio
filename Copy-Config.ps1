# Copy-TraefikConfig.ps1
# Run this from your project root (so .\traefik\* paths resolve)

# List of files to include
$files = @(
    ".\auto_transcribe_audio.py",
    ".\setup_environment.py",
    ".\summarize_transcript.py",
    ".\transcribe_audio.py",
    ".\transcription_config.ini"
)

# Build up the combined content
$combined = ""
foreach ($file in $files) {
    if (Test-Path $file) {
        $combined += "===== $file =====`r`n"
        $combined += Get-Content $file -Raw
        $combined += "`r`n`r`n"
    }
    else {
        $combined += "!!! MISSING: $file !!!`r`n`r`n"
    }
}

# Copy to clipboard
$combined | Set-Clipboard

Write-Host "All configuration files copied to clipboard." -ForegroundColor Green
