# Get the current directory (Project Root)
$projectRoot = Get-Location

# Set PYTHONPATH so Python can find the 'src' folder
$env:PYTHONPATH = "$projectRoot;$env:PYTHONPATH"

Write-Host "Environment set. Project Root: $projectRoot" -ForegroundColor Green
Write-Host "Starting training script..." -ForegroundColor Cyan

# Execute training
python src/train.py --config configs/default.yaml
