"""
Exercise 2.4(b): MIMO Channel GAN Implementation

This script models the MIMO channel generated in Exercise 2.4(a)
using a conditional GAN framework adapted from the original Rayleigh/SISO version.

Expected input file:
    mimo_channel_dataset.mat

Expected variables inside .mat:
    h_mimo : shape [Nr, Nt, Nsnap], e.g. [2, 4, 20000]

Main idea:
    1. Randomly sample one MIMO channel snapshot H from h_mimo
    2. Randomly generate a transmit vector x using QAM symbols
    3. Simulate y = Hx + n
    4. Use real/imag parts of y as real samples
    5. Use real/imag parts of x and H as conditioning vector
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Change this if needed. Use '' or comment out if you want default GPU/CPU behavior.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tf.set_random_seed(100)
np.random.seed(100)

# Training / data configuration
MAT_FILE_PATH = "mimo_channel_dataset.mat"
MODEL_NAME = "ChannelGAN_MIMO"
SAVE_FIG_PATH = MODEL_NAME + "_images"
SAVE_MODEL_PATH = "./Models"

DATA_SIZE = 10000
BATCH_SIZE = 512
Z_DIM = 16
TRAIN_ITERS = 10000     
D_STEPS = 10
PLOT_EVERY = 1000
NOISE_VAR = 0.01

# MIMO dimensions from Exercise 2.4(a)
NUM_RX = 2
NUM_TX = 4

# Network dimensions
OUTPUT_DIM = 2 * NUM_RX                          # [Re(y1), Re(y2), Im(y1), Im(y2)] => 4
CONDITION_DIM = 2 * NUM_TX + 2 * NUM_RX * NUM_TX  # Re/Im(x) + Re/Im(H) => 8 + 16 = 24

# 16-QAM constellation
MEAN_SET_QAM = np.asarray([
    -3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j,
    -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
     1 - 3j,  1 - 1j,  1 + 1j,  1 + 3j,
     3 - 3j,  3 - 1j,  3 + 1j,  3 + 3j
], dtype=np.complex64)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def sample_Z(sample_size):
    """Sample generator input noise from a normal distribution."""
    return np.random.normal(size=sample_size).astype(np.float32)


def xavier_init(size):
    """Xavier initialization."""
    in_dim = size[0]
    xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.0)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
def load_mimo_dataset(mat_file_path):
    """
    Load h_mimo from .mat file.

    Returns:
        h_dataset: complex ndarray with shape [Nr, Nt, Nsnap]
    """
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(
            "Cannot find '{}'. Please run Exercise 2.4(a) first to generate "
            "'mimo_channel_dataset.mat'.".format(mat_file_path)
        )

    mat_data = sio.loadmat(mat_file_path)

    if "h_mimo" not in mat_data:
        raise KeyError(
            "The file '{}' does not contain 'h_mimo'. "
            "Please check your MATLAB save() step.".format(mat_file_path)
        )

    h_dataset = mat_data["h_mimo"]

    if h_dataset.ndim != 3:
        raise ValueError(
            "Expected h_mimo to have 3 dimensions [Nr, Nt, Nsnap], "
            "but got shape {}.".format(h_dataset.shape)
        )

    return h_dataset


def generate_real_samples_with_labels_Rayleigh(h_dataset, number=100, noise_var=0.01):
    """
    MIMO version of the original Rayleigh data generator.

    Args:
        h_dataset: MIMO channel dataset with shape [Nr, Nt, Nsnap]
        number: Number of samples to generate
        noise_var: Noise variance for AWGN

    Returns:
        received_data: shape [number, 2*Nr]
            Example for Nr=2:
            [Re(y1), Re(y2), Im(y1), Im(y2)]

        conditioning: shape [number, 2*Nt + 2*Nr*Nt]
            [Re(x), Im(x), Re(H_flat), Im(H_flat)]
    """
    Nr, Nt, Nsnap = h_dataset.shape

    received_data_list = []
    conditioning_list = []

    for _ in range(number):
        # 1) Randomly select one MIMO channel snapshot H
        idx = np.random.choice(Nsnap)
        H = h_dataset[:, :, idx]    # shape: [Nr, Nt], complex

        # 2) Generate one random transmit vector x from QAM symbols
        symbol_idx = np.random.choice(len(MEAN_SET_QAM), Nt)
        x = MEAN_SET_QAM[symbol_idx].reshape(Nt, 1)   # shape: [Nt, 1]

        # 3) Simulate received signal y = Hx + n
        noise = np.sqrt(noise_var / 2.0) * (
            np.random.randn(Nr, 1) + 1j * np.random.randn(Nr, 1)
        )
        y = H @ x + noise   # shape: [Nr, 1]

        # 4) Construct real sample
        #    [Re(y1), Re(y2), ..., Im(y1), Im(y2), ...]
        y_vec = np.hstack([
            np.real(y).flatten(),
            np.imag(y).flatten()
        ]).astype(np.float32)

        # 5) Construct conditioning vector
        #    [Re(x), Im(x), Re(H_flat), Im(H_flat)]
        H_flat = H.flatten()
        cond_vec = np.hstack([
            np.real(x).flatten(),
            np.imag(x).flatten(),
            np.real(H_flat).flatten(),
            np.imag(H_flat).flatten()
        ]).astype(np.float32)

        # Keep same spirit as original code: mild normalization
        cond_vec = cond_vec / 3.0

        received_data_list.append(y_vec)
        conditioning_list.append(cond_vec)

    received_data = np.asarray(received_data_list, dtype=np.float32)
    conditioning = np.asarray(conditioning_list, dtype=np.float32)

    return received_data, conditioning


# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------
def generator_conditional(z, conditioning):
    """Generator network."""
    z_combine = tf.concat([z, conditioning], axis=1)
    G_h1 = tf.nn.relu(tf.matmul(z_combine, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
    G_logit = tf.matmul(G_h3, G_W4) + G_b4
    return G_logit


def discriminator_conditional(X, conditioning):
    """Discriminator network."""
    x_combine = tf.concat([X, conditioning], axis=1)
    D_h1 = tf.nn.relu(tf.matmul(x_combine / 4.0, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_h3 = tf.nn.relu(tf.matmul(D_h2, D_W3) + D_b3)
    D_logit = tf.matmul(D_h3, D_W4) + D_b4
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit


# -----------------------------------------------------------------------------
# Build graph
# -----------------------------------------------------------------------------
# Discriminator parameters
D_W1 = tf.Variable(xavier_init([OUTPUT_DIM + CONDITION_DIM, 32]))
D_b1 = tf.Variable(tf.zeros(shape=[32]))
D_W2 = tf.Variable(xavier_init([32, 32]))
D_b2 = tf.Variable(tf.zeros(shape=[32]))
D_W3 = tf.Variable(xavier_init([32, 32]))
D_b3 = tf.Variable(tf.zeros(shape=[32]))
D_W4 = tf.Variable(xavier_init([32, 1]))
D_b4 = tf.Variable(tf.zeros(shape=[1]))
theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3, D_W4, D_b4]

# Generator parameters
G_W1 = tf.Variable(xavier_init([Z_DIM + CONDITION_DIM, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))
G_W2 = tf.Variable(xavier_init([128, 128]))
G_b2 = tf.Variable(tf.zeros(shape=[128]))
G_W3 = tf.Variable(xavier_init([128, 128]))
G_b3 = tf.Variable(tf.zeros(shape=[128]))
G_W4 = tf.Variable(xavier_init([128, OUTPUT_DIM]))
G_b4 = tf.Variable(tf.zeros(shape=[OUTPUT_DIM]))
theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3, G_W4, G_b4]

# Placeholders
R_sample = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM])
Z = tf.placeholder(tf.float32, shape=[None, Z_DIM])
Condition = tf.placeholder(tf.float32, shape=[None, CONDITION_DIM])

# Forward pass
G_sample = generator_conditional(Z, Condition)
D_prob_real, D_logit_real = discriminator_conditional(R_sample, Condition)
D_prob_fake, D_logit_fake = discriminator_conditional(G_sample, Condition)

# WGAN-GP losses
D_loss = tf.reduce_mean(D_logit_fake) - tf.reduce_mean(D_logit_real)
G_loss = -tf.reduce_mean(D_logit_fake)

lambda_gp = 5.0
alpha = tf.random_uniform(shape=tf.shape(R_sample), minval=0.0, maxval=1.0)
differences = G_sample - R_sample
interpolates = R_sample + alpha * differences
_, D_inter = discriminator_conditional(interpolates, Condition)
gradients = tf.gradients(D_inter, [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-8)
gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
D_loss += lambda_gp * gradient_penalty

D_solver = tf.train.AdamOptimizer(
    learning_rate=1e-4, beta1=0.5, beta2=0.9
).minimize(D_loss, var_list=theta_D)

G_solver = tf.train.AdamOptimizer(
    learning_rate=1e-4, beta1=0.5, beta2=0.9
).minimize(G_loss, var_list=theta_G)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def plot_real_distribution(data, save_fig_path):
    """
    Plot Rx antenna 1 constellation-like view:
    x-axis = Re(y1), y-axis = Im(y1)
    """
    plt.figure(figsize=(5, 5))
    plt.plot(data[:1000, 0], data[:1000, NUM_RX], "b.", alpha=0.6)
    plt.xlabel(r"$Re\{y_1\}$")
    plt.ylabel(r"$Im\{y_1\}$")
    plt.title("Real data distribution of Rx antenna 1")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_fig_path, "real_rx1.png"), bbox_inches="tight")
    plt.close()


def build_plot_conditioning_from_fixed_channel(H, number=20):
    """
    Build conditioning vectors for plotting generated samples using one fixed MIMO channel.
    Only the first Tx stream is swept over QAM labels for visualization simplicity.
    Other Tx streams are set to zero.

    Returns:
        conditioning: [16*number, CONDITION_DIM]
    """
    H_flat = H.flatten()
    conditioning_list = []

    for qam_symbol in MEAN_SET_QAM:
        for _ in range(number):
            x = np.zeros((NUM_TX, 1), dtype=np.complex64)
            x[0, 0] = qam_symbol

            cond_vec = np.hstack([
                np.real(x).flatten(),
                np.imag(x).flatten(),
                np.real(H_flat).flatten(),
                np.imag(H_flat).flatten()
            ]).astype(np.float32) / 3.0

            conditioning_list.append(cond_vec)

    return np.asarray(conditioning_list, dtype=np.float32)


def plot_generated_samples(sess, G_sample_tensor, Z_tensor, Condition_tensor, h_dataset, save_fig_path, step):
    """
    Plot generated samples for one randomly selected fixed channel.
    Visualize Rx antenna 1 only:
        x-axis = Re(y1_hat), y-axis = Im(y1_hat)
    """
    idx = np.random.choice(h_dataset.shape[2])
    H = h_dataset[:, :, idx]

    conditioning = build_plot_conditioning_from_fixed_channel(H, number=20)
    z_input = sample_Z((conditioning.shape[0], Z_DIM))
    samples = sess.run(
        G_sample_tensor,
        feed_dict={Z_tensor: z_input, Condition_tensor: conditioning}
    )

    plt.figure(figsize=(5, 5))
    plt.plot(samples[:, 0], samples[:, NUM_RX], "r.", alpha=0.6)
    plt.xlabel(r"$Re\{\hat{y}_1\}$")
    plt.ylabel(r"$Im\{\hat{y}_1\}$")
    plt.title("Generated samples of Rx antenna 1 at step {}".format(step))
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(save_fig_path, "generated_rx1_step_{:06d}.png".format(step)),
        bbox_inches="tight"
    )
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ensure_dir(SAVE_FIG_PATH)
    ensure_dir(SAVE_MODEL_PATH)

    # Load dataset
    h_dataset = load_mimo_dataset(MAT_FILE_PATH)
    print("Loaded h_dataset shape =", h_dataset.shape)

    if h_dataset.shape[0] != NUM_RX or h_dataset.shape[1] != NUM_TX:
        raise ValueError(
            "Expected h_dataset shape to start with ({}, {}), but got {}.".format(
                NUM_RX, NUM_TX, h_dataset.shape
            )
        )

    # Build training data once
    data, one_hot_labels = generate_real_samples_with_labels_Rayleigh(
        h_dataset, number=DATA_SIZE, noise_var=NOISE_VAR
    )
    print("data shape =", data.shape)
    print("conditioning shape =", one_hot_labels.shape)

    # Save a quick view of real data
    plot_real_distribution(data, SAVE_FIG_PATH)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    for it in range(TRAIN_ITERS):
        start_idx = (it * BATCH_SIZE) % DATA_SIZE

        if start_idx + BATCH_SIZE >= len(data):
            # reshuffle / regenerate mini-batch source once a cycle is over
            perm = np.random.permutation(DATA_SIZE)
            data = data[perm]
            one_hot_labels = one_hot_labels[perm]
            start_idx = 0

        X_mb = data[start_idx:start_idx + BATCH_SIZE, :]
        cond_mb = one_hot_labels[start_idx:start_idx + BATCH_SIZE, :]

        # Train discriminator multiple times
        for _ in range(D_STEPS):
            _, D_loss_curr = sess.run(
                [D_solver, D_loss],
                feed_dict={
                    R_sample: X_mb,
                    Z: sample_Z((BATCH_SIZE, Z_DIM)),
                    Condition: cond_mb
                }
            )

        # Train generator once
        _, G_loss_curr = sess.run(
            [G_solver, G_loss],
            feed_dict={
                R_sample: X_mb,
                Z: sample_Z((BATCH_SIZE, Z_DIM)),
                Condition: cond_mb
            }
        )

        if (it + 1) % 100 == 0:
            print("Iter: {:6d} | D_loss: {:8.4f} | G_loss: {:8.4f}".format(
                it + 1, D_loss_curr, G_loss_curr
            ))

        if (it + 1) % PLOT_EVERY == 0:
            ckpt_path = saver.save(
                sess,
                os.path.join(SAVE_MODEL_PATH, "ChannelGAN_MIMO_step_{}.ckpt".format(it + 1))
            )
            print("Model saved to:", ckpt_path)

            plot_generated_samples(
                sess, G_sample, Z, Condition, h_dataset, SAVE_FIG_PATH, it + 1
            )

    # Final save
    ckpt_path = saver.save(
        sess,
        os.path.join(SAVE_MODEL_PATH, "ChannelGAN_MIMO_final.ckpt")
    )
    print("Final model saved to:", ckpt_path)

    sess.close()


if __name__ == "__main__":
    main()