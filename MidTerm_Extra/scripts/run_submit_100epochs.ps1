$ErrorActionPreference = "Stop"

$py = "C:\Users\York\.conda\envs\csinet_tf\python.exe"

if (-not (Test-Path $py)) {
  throw "Python not found at $py"
}

Write-Host "Working directory: $PWD"
Write-Host "Using Python: $py"

Write-Host "`n[0/7] Cleaning previous submit outputs..."
& "$PSScriptRoot\clean_submit_outputs.ps1"

Write-Host "`n[1/7] Generating synthetic time-varying CSI sequences..."
& $py src/data/make_time_sequence.py `
  --input-dir ../MidTerm_Q7/data/cost2100_official `
  --output-dir data/sequence_submit `
  --time-steps 10 `
  --rho-list 0.95 0.80 0.50 `
  --train-limit 100 `
  --val-limit 50 `
  --test-limit 50

Write-Host "`n[2/7] Training DA-TCFNet for 100 epochs..."
& $py -u src/train_proposed.py `
  --data-dir data/sequence_submit `
  --time-steps 10 `
  --encoded-dim 128 `
  --epochs 100 `
  --batch-size 64 `
  --save-dir saved_model/proposed_submit `
  --result-dir result/submit

Write-Host "`n[3/7] Evaluating DA-TCFNet on the test split..."
& $py -u src/test_proposed.py `
  --data-file data/sequence_submit/test_sequences_t10.npz `
  --weights saved_model/proposed_submit/da_tcfnet_best.weights.h5 `
  --time-steps 10 `
  --encoded-dim 128 `
  --batch-size 64

Write-Host "`n[4/7] Running three-way ablation for 100 epochs..."
& $py -u src/run_ablation.py `
  --data-dir data/sequence_submit `
  --time-steps 10 `
  --encoded-dim 128 `
  --epochs 100 `
  --batch-size 64 `
  --result-dir result/ablation_submit_triplet

Write-Host "`n[5/7] Plotting ablation and training-curve figures..."
& $py -u src/utils/plot_ablation_results.py `
  --ablation-csv result/ablation_submit_triplet/ablation_results.csv `
  --history-csv result/submit/history_da_tcfnet.csv `
  --figure-dir result/figures_submit

Write-Host "`n[6/7] Plotting data-overview figures..."
& $py -u src/utils/plot_data_overview.py `
  --data-dir data/sequence_submit `
  --time-steps 10 `
  --figure-dir result/figures_submit

Write-Host "`n[7/7] Plotting reconstruction comparison figures..."
& $py -u src/utils/plot_reconstruction_examples.py `
  --data-file data/sequence_submit/test_sequences_t10.npz `
  --model-dir result/ablation_submit_triplet/models `
  --time-steps 10 `
  --encoded-dim 128 `
  --rho-values 0.95 0.50 `
  --figure-dir result/figures_submit

Write-Host "`nFinished full submit pipeline."
