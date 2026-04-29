"""
CsiNet Inference Only for CSI Compression and Reconstruction
This script loads a pre-trained CsiNet autoencoder model (architecture + weights)
to perform only inference on test CSI data for indoor/outdoor wireless environments.
It evaluates model performance with NMSE (Normalized Mean Square Error) and correlation coefficient,
measures per-sample inference time, and visualizes the original vs. reconstructed CSI amplitude.
No model training is performed in this script.
"""
import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from keras.models import Model, model_from_json
from keras.callbacks import TensorBoard, Callback
import scipy.io as sio 
import numpy as np
import math
import time
tf.reset_default_graph()
# ────────────────────────  Environment & CSI Configuration  ───────────────────────── #
envir = 'indoor' #'indoor' or 'outdoor' -> Select the wireless propagation environment
# ────────────────────────  CSI Image Parameters  ────────────────────────────── #
img_height = 32        # CSI matrix height (spatial dimension)
img_width = 32         # CSI matrix width (frequency dimension)
img_channels = 2       # Real and imaginary parts of CSI (2 separate channels)
img_total = img_height*img_width*img_channels  # Total flattened CSI feature dimensions
# ────────────────────────  Network Hyperparameters  ─────────────────────────── #
residual_num = 2       # Number of residual blocks in the decoder (match pre-trained model)
encoded_dim = 512      # Compression dimension (match pre-trained model): 1/4→512,1/16→128,1/32→64,1/64→32
# Generate model file name (consistent with training script naming convention)
file = 'CsiNet_'+(envir)+'_dim'+str(encoded_dim)
# ────────────────────────  Load Pre-trained CsiNet Model  ────────────────────── #
# Load model architecture from JSON file
outfile = "saved_model/model_%s.json"%file
json_file = open(outfile, 'r')
loaded_model_json = json_file.read()
json_file.close()
autoencoder = model_from_json(loaded_model_json)  # Reconstruct model from JSON

# Load pre-trained model weights from HDF5 file
outfile = "saved_model/model_%s.h5"%file
autoencoder.load_weights(outfile)  # Load weights into the reconstructed model
# ────────────────────────  Test Data Loading and Preprocessing  ─────────────── #
# Load MATLAB-formatted test CSI data for the selected environment
if envir == 'indoor':
    mat = sio.loadmat('data/DATA_Htestin.mat')
    x_test = mat['HT'] # Indoor test CSI data array
elif envir == 'outdoor':
    mat = sio.loadmat('data/DATA_Htestout.mat')
    x_test = mat['HT'] # Outdoor test CSI data array

# Convert data to float32 (matching training data type for inference)
x_test = x_test.astype('float32')
# Reshape to channels_first format: [batch_size, channels, height, width] (match model input)
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))
# ────────────────────────  CSI Reconstruction Inference  ────────────────────── #
# Measure total inference time for the entire test set
tStart = time.time()
x_hat = autoencoder.predict(x_test)  # Reconstruct CSI from test input (forward pass only)
tEnd = time.time()
# Calculate and print average inference time per test sample
print ("It cost %f sec" % ((tEnd - tStart)/x_test.shape[0]))
# ────────────────────────  Performance Evaluation (NMSE & Correlation)  ─────── #
# Load original frequency-domain CSI data for correlation coefficient calculation
if envir == 'indoor':
    mat = sio.loadmat('data/DATA_HtestFin_all.mat')
    X_test = mat['HF_all']# Indoor frequency-domain CSI array
elif envir == 'outdoor':
    mat = sio.loadmat('data/DATA_HtestFout_all.mat')
    X_test = mat['HF_all']# Outdoor frequency-domain CSI array

# Reshape frequency-domain CSI to [batch_size, img_height, 125] (original frequency bins)
X_test = np.reshape(X_test, (len(X_test), img_height, 125))

# Convert original CSI from [0,1] tensor to complex domain (-0.5~0.5 for real/imag parts)
x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)

# Convert reconstructed CSI from [0,1] tensor to complex domain (same as original)
x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)

# Reshape complex CSI to frequency domain and perform FFT for correlation calculation
x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
# Zero-padding to 257 frequency bins + FFT + truncate to original 125 bins (matching X_test)
X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257-img_width))), axis=2), axis=2)
X_hat = X_hat[:, :, 0:125]

# Calculate correlation coefficient (rho) between original and reconstructed frequency-domain CSI
n1 = np.sqrt(np.sum(np.conj(X_test)*X_test, axis=1))  # L2 norm of original frequency-domain CSI
n1 = n1.astype('float64')
n2 = np.sqrt(np.sum(np.conj(X_hat)*X_hat, axis=1))  # L2 norm of reconstructed frequency-domain CSI
n2 = n2.astype('float64')
aa = abs(np.sum(np.conj(X_test)*X_hat, axis=1))     # Cross inner product for correlation
rho = np.mean(aa/(n1*n2), axis=1)                   # Per-sample correlation coefficient

# Reshape for NMSE (Normalized Mean Square Error) calculation (flatten to 1D)
X_hat = np.reshape(X_hat, (len(X_hat), -1))
X_test = np.reshape(X_test, (len(X_test), -1))

# Compute NMSE (in dB) for time-domain CSI reconstruction
power = np.sum(abs(x_test_C)**2, axis=1)    # Power of original complex CSI
mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)  # MSE between original and reconstructed complex CSI
# Print key performance metrics for the selected environment and compression rate
print("In "+envir+" environment")
print("When dimension is", encoded_dim)
print("NMSE is ", 10*math.log10(np.mean(mse/power)))  # NMSE in logarithmic dB scale
print("Correlation is ", np.mean(rho))                # Average correlation coefficient over test set
# ────────────────────────  CSI Reconstruction Visualization  ────────────────── #
import matplotlib.pyplot as plt
'''Plot absolute amplitude of original and reconstructed complex CSI (first 10 test samples)'''
n = 10  # Number of samples to visualize
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original CSI absolute amplitude
    ax = plt.subplot(2, n, i + 1 )
    x_testplo = abs(x_test[i, 0, :, :]-0.5 + 1j*(x_test[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)  # Invert for better visualization
    plt.gray()  # Grayscale colormap
    ax.get_xaxis().set_visible(False)  # Hide x-axis
    ax.get_yaxis().set_visible(False)  # Hide y-axis
    ax.invert_yaxis()  # Invert y-axis for consistent spatial alignment

    # Display reconstructed CSI absolute amplitude
    ax = plt.subplot(2, n, i + 1 + n)
    decoded_imgsplo = abs(x_hat[i, 0, :, :]-0.5 
                          + 1j*(x_hat[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)  # Invert for better visualization
    plt.gray()  # Grayscale colormap
    ax.get_xaxis().set_visible(False)  # Hide x-axis
    ax.get_yaxis().set_visible(False)  # Hide y-axis
    ax.invert_yaxis()  # Invert y-axis for consistent spatial alignment
plt.show()  # Show the visualization plot