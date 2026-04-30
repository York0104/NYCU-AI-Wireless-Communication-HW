conda activate csinet_tf
python src/data/make_time_sequence.py `
  --input-dir ../MidTerm_Q7/data/cost2100_official `
  --output-dir data/sequence_submit `
  --time-steps 10 `
  --rho-list 0.95 0.80 0.50 `
  --train-limit 100 `
  --val-limit 50 `
  --test-limit 50
