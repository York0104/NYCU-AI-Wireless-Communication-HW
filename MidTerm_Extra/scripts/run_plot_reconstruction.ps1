conda activate csinet_tf
python src/utils/plot_reconstruction_examples.py `
  --data-file data/sequence_submit/test_sequences_t10.npz `
  --model-dir result/ablation_submit_triplet/models `
  --time-steps 10 `
  --encoded-dim 128 `
  --rho-values 0.95 0.50 `
  --figure-dir result/figures_submit
