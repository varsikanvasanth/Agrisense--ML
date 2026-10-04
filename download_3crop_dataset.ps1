$ErrorActionPreference = "Stop"
$dest = Join-Path $PSScriptRoot "data\PlantVillage-3Crop"
if (Test-Path $dest) { Write-Host "Dataset folder already exists: $dest"; exit }
Write-Host "Downloading only Tomato + Corn/Maize + Potato folders from PlantVillage GitHub..."
git clone --filter=blob:none --sparse https://github.com/gabrieldgf4/PlantVillage-Dataset.git $dest
Set-Location $dest
git sparse-checkout set "Tomato___*" "Corn_(maize)___*" "Potato___*"
Write-Host "3-crop dataset download complete."
Write-Host "Now copy the class folders into: $PSScriptRoot\data\plantvillage"
