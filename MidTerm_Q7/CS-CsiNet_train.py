"""
CS-CsiNet Training for CSI Compression and Reconstruction
This script implements the Compressed Sensing based CsiNet (CS-CsiNet) for Channel State Information (CSI) recovery,
adopting a **fixed random projection encoder** + **trainable residual-based decoder** architecture.
It trains the decoder on indoor/outdoor wireless CSI datasets, supports adjustable compression rates,
evaluates performance with NMSE (Normalized Mean Square Error) and correlation coefficient,
and saves training logs, model weights, reconstruction results and visualization plots.
"""
import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from keras.models import Model
from keras.callbacks import TensorBoard, Callback
import scipy.io as sio 
import numpy as np
import math
import time
tf.reset_default_graph()
# ────────────────────────  Environment Configuration  ───────────────────────── #
envir = 'indoor' #'indoor' or 'outdoor' -> Select wireless propagation environment
# ────────────────────────  CSI Image Parameters  ────────────────────────────── #
img_height = 32        # CSI matrix height (spatial dimension)
img_width = 32         # CSI matrix width (frequency dimension)
img_channels = 2       # Real and imaginary parts of CSI (2 separate channels)
img_total = img_height*img_width*img_channels  # Total flattened CSI feature dimensions
# ────────────────────────  Network Hyperparameters  ─────────────────────────── #
residual_num = 2       # Number of residual blocks in the trainable decoder
encoded_dim = 512      # Compression dimension: 1/4→512, 1/16→128, 1/32→64, 1/64→32
# ────────────────────────  CS-CsiNet Decoder Construction  ─────────────────── #
def residual_network(encoded, residual_num, encoded_dim):
    """
    Build the residual-based trainable decoder for CS-CsiNet.
    Input: Compressed CSI feature vector (from random projection encoder)
    Output: Reconstructed CSI tensor (shape: [img_channels, img_height, img_width])
    Args:
        encoded (tensor): Compressed CSI feature vector (shape: [encoded_dim,])
        residual_num (int): Number of residual blocks in the decoder
        encoded_dim (int): Dimension of the compressed CSI feature vector
    Returns:
        tensor: Reconstructed CSI tensor with sigmoid activation (range [0,1])
    """
    def add_common_layers(y):
        """Add shared BatchNormalization and LeakyReLU activation for residual blocks."""
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        return y

    def residual_block_decoded(y):
        """Residual block for decoder: 3x3 Conv2D stack with shortcut skip connection."""
        shortcut = y  # Shortcut connection for residual learning
        y = Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = add_common_layers(y)
        
        y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = add_common_layers(y)
        
        y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = BatchNormalization()(y)
        y = add([shortcut, y])  # Residual fusion: shortcut + conv stack output
        y = LeakyReLU()(y)
        return y
    
    # Decoder pipeline: Dense decompression → Reshape → Residual blocks → Final Conv2D
    decoded = Dense(img_total, activation='linear')(encoded)  # Decompress to flattened CSI dim
    decoded = Reshape((img_channels, img_height, img_width,))(decoded)  # Reshape to 4D CSI tensor
    for i in range(residual_num):
        decoded = residual_block_decoded(decoded)  # Stack residual blocks for high-quality reconstruction
    
    decoded = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(decoded)
    return decoded

# Build CS-CsiNet model: Input=compressed vector, Output=reconstructed CSI
image_tensor = Input(shape=(encoded_dim, ))  # Input shape: compressed CSI feature vector
network_output = residual_network(image_tensor, residual_num, encoded_dim)  # Decoder output
decoder = Model(inputs=[image_tensor], outputs=[network_output])  # Define trainable decoder
decoder.compile(optimizer='adam', loss='mse')  # Compile with Adam & MSE loss (CSI reconstruction)
print(decoder.summary())  # Print decoder architecture and parameter count
# ────────────────────────  Data Loading and Preprocessing  ──────────────────── #
# Load MATLAB-formatted CSI datasets (train/val/test) for selected environment
if envir == 'indoor':
    mat = sio.loadmat('data/DATA_Htrainin.mat') 
    x_train = mat['HT'] # Indoor training CSI data array
    mat = sio.loadmat('data/DATA_Hvalin.mat')
    x_val = mat['HT'] # Indoor validation CSI data array
    mat = sio.loadmat('data/DATA_Htestin.mat')
    x_test = mat['HT'] # Indoor test CSI data array
elif envir == 'outdoor':
    mat = sio.loadmat('data/DATA_Htrainout.mat') 
    x_train = mat['HT'] # Outdoor training CSI data array
    mat = sio.loadmat('data/DATA_Hvalout.mat')
    x_val = mat['HT'] # Outdoor validation CSI data array
    mat = sio.loadmat('data/DATA_Htestout.mat')
    x_test = mat['HT'] # Outdoor test CSI data array

# Convert data to float32 for neural network training/inference
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

# ────────────────────────  Fixed Random Projection Encoder  ─────────────────── #
# Load pre-defined random projection matrix A (CS encoder) from MAT file
mat = sio.loadmat('data/A%d.mat'%(encoded_dim))
A = mat['A'] # Random projection matrix (fixed, non-trainable)
# Perform CS compression: project original CSI to low-dimensional feature vector
y_train = np.dot(x_train, A.T)  # Compressed training data
y_val = np.dot(x_val, A.T)      # Compressed validation data
y_test = np.dot(x_test, A.T)    # Compressed test data

# Reshape original CSI data to channels_first format [batch, channels, height, width] (decoder target)
x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))
x_val = np.reshape(x_val, (len(x_val), img_channels, img_height, img_width))
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))
# ────────────────────────  Custom Loss Callback  ─────────────────────────────── #
class LossHistory(Callback):
    """Custom Keras Callback to record batch-wise training loss and epoch-wise validation loss."""
    def on_train_begin(self, logs={}):
        """Initialize empty loss lists at the start of training."""
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        """Append training loss of the current batch to the training loss list."""
        self.losses_train.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs={}):
        """Append validation loss of the current epoch to the validation loss list."""
        self.losses_val.append(logs.get('val_loss'))
        
# Initialize the custom loss history callback
history = LossHistory()
# ────────────────────────  Model Training  ──────────────────────────────────── #
# Generate unique file name with environment, compression dim and current date
file = 'CS-CsiNet_'+(envir)+'_dim'+str(encoded_dim)+time.strftime('_%m_%d')
path = 'result/TensorBoard_%s' %file  # TensorBoard log directory for training visualization

# Train the CS-CsiNet decoder (input=compressed CSI, target=original CSI)
decoder.fit(y_train, x_train,
            epochs=1000,               # Total training epochs
            batch_size=200,            # Mini-batch size for training
            shuffle=True,              # Shuffle training data per epoch
            validation_data=(y_val, x_val),  # Validation dataset (compressed → original)
            callbacks=[history,        # Record training/validation loss
                       TensorBoard(log_dir = path)])  # TensorBoard training logs

# Save training and validation loss to CSV files for post-analysis
filename = 'result/trainloss_%s.csv'%file
loss_history = np.array(history.losses_train)
np.savetxt(filename, loss_history, delimiter=",")

filename = 'result/valloss_%s.csv'%file
loss_history = np.array(history.losses_val)
np.savetxt(filename, loss_history, delimiter=",")
# ────────────────────────  Model Inference on Test Data  ────────────────────── #
# Measure total inference time for CSI reconstruction on the test set
tStart = time.time()
x_hat = decoder.predict(y_test)  # Reconstruct CSI from compressed test data
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

# Convert reconstructed CSI from [0,1] tensor to complex domain (matching original CSI)
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
aa = abs(np.sum(np.conj(X_test)*X_hat, axis=1))     # Cross inner product for correlation calculation
rho = np.mean(aa/(n1*n2), axis=1)                   # Per-sample correlation coefficient

# Reshape for NMSE (Normalized Mean Square Error) calculation (flatten to 1D vector)
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

# Save reconstructed CSI and correlation coefficient to CSV files for post-processing
filename = "result/decoded_%s.csv"%file
x_hat1 = np.reshape(x_hat, (len(x_hat), -1))
np.savetxt(filename, x_hat1, delimiter=",")

filename = "result/rho_%s.csv"%file
np.savetxt(filename, rho, delimiter=",")
# ────────────────────────  CSI Reconstruction Visualization  ────────────────── #
import matplotlib.pyplot as plt
'''Plot absolute amplitude of original and reconstructed complex CSI (first 10 test samples)'''
n = 10  # Number of test samples to visualize
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original CSI absolute amplitude
    ax = plt.subplot(2, n, i + 1 )
    x_testplo = abs(x_test[i, 0, :, :]-0.5 + 1j*(x_test[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)  # Invert for better visual contrast
    plt.gray()  # Use grayscale colormap for CSI amplitude visualization
    ax.get_xaxis().set_visible(False)  # Hide x-axis for clean visualization
    ax.get_yaxis().set_visible(False)  # Hide y-axis for clean visualization
    ax.invert_yaxis()  # Invert y-axis for consistent spatial alignment of CSI matrix

    # Display reconstructed CSI absolute amplitude
    ax = plt.subplot(2, n, i + 1 + n)
    decoded_imgsplo = abs(x_hat[i, 0, :, :]-0.5 
                          + 1j*(x_hat[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)  # Invert for better visual contrast
    plt.gray()  # Use grayscale colormap for CSI amplitude visualization
    ax.get_xaxis().set_visible(False)  # Hide x-axis for clean visualization
    ax.get_yaxis().set_visible(False)  # Hide y-axis for clean visualization
    ax.invert_yaxis()  # Invert y-axis for consistent spatial alignment of CSI matrix
plt.show()  # Show the original vs. reconstructed CSI visualization plot
# ────────────────────────  Model Saving  ────────────────────────────────────── #
# Serialize decoder model architecture to JSON file (for inference)
model_json = decoder.to_json()
outfile = "result/model_%s.json"%file
with open(outfile, "w") as json_file:
    json_file.write(model_json)
# Serialize decoder model weights to HDF5 file (pre-trained weights for inference)
outfile = "result/model_%s.h5"%file
decoder.save_weights(outfile)