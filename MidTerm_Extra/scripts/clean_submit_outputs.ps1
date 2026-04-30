$resultPaths = @(
  "data\sequence_submit",
  "result\submit",
  "result\ablation_submit_triplet",
  "result\figures_submit",
  "saved_model\proposed_submit"
)

foreach ($path in $resultPaths) {
  if (Test-Path $path) {
    Remove-Item -Recurse -Force $path
    Write-Host "Removed $path"
  } else {
    Write-Host "Skip missing $path"
  }
}
