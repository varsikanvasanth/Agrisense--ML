$ErrorActionPreference = "Stop"
$src = Join-Path $PSScriptRoot "data\PlantVillage-3Crop"
$dst = Join-Path $PSScriptRoot "data\plantvillage"
New-Item -ItemType Directory -Force $dst | Out-Null
$patterns = @("Tomato___*","Corn_(maize)___*","Potato___*")
foreach ($pat in $patterns) {
  Get-ChildItem -Path $src -Directory -Filter $pat | ForEach-Object {
    Copy-Item $_.FullName -Destination $dst -Recurse -Force
  }
}
Write-Host "Prepared 3-crop dataset in $dst"
Get-ChildItem $dst -Directory | Select-Object Name
