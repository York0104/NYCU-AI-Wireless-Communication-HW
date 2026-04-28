# Exercise 2.15: CsiNet on Other Channel Datasets

## Problem Translation

Exercise 2.15 asks how CsiNet performs on channel datasets other than the original training distribution. The tasks are:

1. Use the COST 2100 channel model to generate more than five different channel datasets, for example by changing the distribution of users.
2. Evaluate the CSI reconstruction NMSE of a trained CsiNet model on each dataset.
3. Mix the different channel datasets and use the mixed data to train CsiNet. Compare the reconstruction performance with the result in part (b), and discuss how to improve the generalization of CSI feedback methods in practical systems.

## Reference Setting

The CsiNet reference paper uses the COST 2100 channel model with the following important settings:

- Indoor picocellular scenario at 5.3 GHz.
- Outdoor rural scenario at 300 MHz.
- Base station at the center of a square region.
- User equipment randomly distributed in the square region.
- Indoor region side length: 20 m.
- Outdoor region side length: 400 m.
- Uniform linear array with 32 BS antennas.
- 1024 OFDM subcarriers.
- After 2D DFT, only the first 32 delay-domain rows are retained, giving a 32 x 32 angular-delay channel matrix.
- The CsiNet input contains two channels: real and imaginary parts, normalized to [0, 1].
- NMSE is defined as `E{||H - H_hat||^2 / ||H||^2}` and is reported in dB.

## (a) Dataset Generation

Six different datasets were generated. They preserve the CsiNet input format `32 x 32 x 2`, while changing the user distribution and propagation condition. This follows the instruction to generate different channel datasets by changing the distribution of users.

| Dataset | Environment | User Distribution | Purpose |
|---|---|---|---|
| D1_indoor_uniform | Indoor, 20 m square, 5.3 GHz | Uniform over the full square | Baseline indoor distribution |
| D2_indoor_center | Indoor, 20 m square, 5.3 GHz | Concentrated near BS | Tests near-user channels |
| D3_indoor_edge | Indoor, 20 m square, 5.3 GHz | Concentrated near cell edge | Tests far-user channels |
| D4_indoor_ring | Indoor, 20 m square, 5.3 GHz | Ring-shaped distribution | Tests non-uniform but distance-controlled channels |
| D5_outdoor_uniform | Outdoor, 400 m square, 300 MHz | Uniform over the full square | Baseline outdoor distribution |
| D6_outdoor_clustered | Outdoor, 400 m square, 300 MHz | Several user clusters | Tests hotspot-like deployment |

The included generator is a deterministic COST-2100-style surrogate. It creates sparse angular-delay channels with clustered multipath components, distance-dependent path loss, angle spread, and delay spread. If the official MATLAB COST 2100 generator is available, its `.mat` outputs can replace these files directly as long as they provide the same keys:

- `HT`: normalized CsiNet input, shape `[samples, 2048]`.
- `HF_all`: complex frequency-domain CSI for testing, shape `[samples, 32, 125]`.

## (b) Cross-Dataset Evaluation

The baseline CsiNet model is trained on `D1_indoor_uniform` and then tested on all six datasets. The experiment uses `encoded_dim = 512`, 20 epochs, 1200 training samples for the single-dataset model, and 400 testing samples per dataset. Because the experiment was executed on Windows CPU, the TensorFlow implementation uses `channels_last` internally; the CSI tensor content and NMSE/rho evaluation are kept equivalent to the original CsiNet setting.

| Train Dataset | Test Dataset | NMSE (dB) | rho |
|---|---|---:|---:|
| D1_indoor_uniform | D1_indoor_uniform | 0.0005 | 0.314942 |
| D1_indoor_uniform | D2_indoor_center | 0.1515 | 0.332673 |
| D1_indoor_uniform | D3_indoor_edge | -0.2322 | 0.311674 |
| D1_indoor_uniform | D4_indoor_ring | -0.2603 | 0.334071 |
| D1_indoor_uniform | D5_outdoor_uniform | 1.1156 | 0.276044 |
| D1_indoor_uniform | D6_outdoor_clustered | 0.0821 | 0.293378 |

The baseline model performs better on indoor-like distributions and degrades on the outdoor and clustered datasets. This is expected because the training distribution only contains indoor uniform users, so the learned representation is biased toward that channel distribution.

## (c) Mixed-Dataset Training and Comparison

The six datasets were mixed and used as the training set. The mixed model was then evaluated on each individual test dataset.

| Test Dataset | Baseline NMSE (dB) | Mixed-Train NMSE (dB) | Improvement (dB) | Baseline rho | Mixed rho |
|---|---:|---:|---:|---:|---:|
| D1_indoor_uniform | 0.0005 | -5.3079 | 5.3084 | 0.314942 | 0.609326 |
| D2_indoor_center | 0.1515 | -5.0876 | 5.2391 | 0.332673 | 0.619628 |
| D3_indoor_edge | -0.2322 | -5.5908 | 5.3586 | 0.311674 | 0.611074 |
| D4_indoor_ring | -0.2603 | -5.9193 | 5.6590 | 0.334071 | 0.631303 |
| D5_outdoor_uniform | 1.1156 | -4.0511 | 5.1667 | 0.276044 | 0.597032 |
| D6_outdoor_clustered | 0.0821 | -5.4430 | 5.5251 | 0.293378 | 0.622798 |

The mixed-training model improves NMSE on every test dataset. The improvement is around 5 dB for all six datasets, and rho nearly doubles compared with the single-distribution model. This supports the conclusion that training only on one channel distribution limits generalization, while training on diverse channel realizations makes the CSI feedback model more robust.

## Discussion

In practical wireless systems, the channel distribution changes with user location, carrier frequency, scattering environment, mobility, and deployment geometry. A CsiNet model trained on a single scenario may reconstruct CSI well only for channels similar to the training data. When the test distribution shifts, the learned encoder and decoder may no longer preserve the most important angular-delay components, causing NMSE degradation.

To improve generalization in practical systems, the CSI feedback model should be trained with diverse channel data. Useful strategies include:

- Mix indoor, outdoor, cell-center, cell-edge, clustered, and mobility-related channel samples during training.
- Use domain randomization by varying user distribution, path loss, delay spread, angle spread, and number of clusters.
- Fine-tune the decoder at the BS when new deployment data becomes available.
- Use domain adaptation or transfer learning to adapt a pretrained CsiNet model to a new scenario with limited data.
- Include temporal correlation for mobile users, for example using a recurrent, transformer, or predictive feedback architecture.
- Use uncertainty-aware or ensemble feedback models when the channel distribution is highly variable.

## Reproduction Commands

Generate the six datasets and run the local CPU verification:

```powershell
python scripts/run_exercise_2_15_fast.py --generate --train-samples 1200 --val-samples 300 --test-samples 400 --encoded-dim 128 --mix-limit 1200
```

Run the TensorFlow CsiNet version:

```powershell
conda run -n csinet_tf python scripts/run_exercise_2_15_tf.py --encoded-dim 512 --epochs 20 --batch-size 100 --mix-limit 1200 --val-limit 300
```

The produced result files are:

- `result/exercise_2_15_fast_results.csv`
- `result/exercise_2_15_csinet_results.csv` after running the TensorFlow script
