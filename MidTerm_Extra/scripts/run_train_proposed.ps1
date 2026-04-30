conda activate csinet_tf
python src/train_proposed.py `
  --data-dir data/sequence_submit `
  --time-steps 10 `
  --encoded-dim 128 `
  --epochs 5 `
  --batch-size 64 `
  --save-dir saved_model/proposed_submit `
  --result-dir result/submit
