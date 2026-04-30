conda activate csinet_tf
python src/run_ablation.py `
  --data-dir data/sequence_submit `
  --time-steps 10 `
  --encoded-dim 128 `
  --epochs 5 `
  --batch-size 64 `
  --result-dir result/ablation_submit_triplet
