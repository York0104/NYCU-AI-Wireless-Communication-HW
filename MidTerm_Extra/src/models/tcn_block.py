from __future__ import annotations


def build_tcn_feature(tf, latent_sequence, filters: int = 256, kernel_size: int = 3, dilation_rates: tuple[int, ...] = (1, 2, 4)):
    layers = tf.keras.layers
    x = latent_sequence
    for idx, dilation in enumerate(dilation_rates):
        shortcut = x
        y = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation, name=f"tcn_{idx}_conv1")(x)
        y = layers.BatchNormalization(name=f"tcn_{idx}_bn1")(y)
        y = layers.LeakyReLU(name=f"tcn_{idx}_act1")(y)
        y = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation, name=f"tcn_{idx}_conv2")(y)
        y = layers.BatchNormalization(name=f"tcn_{idx}_bn2")(y)
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, padding="same", name=f"tcn_{idx}_proj")(shortcut)
        x = layers.Add(name=f"tcn_{idx}_add")([shortcut, y])
        x = layers.LeakyReLU(name=f"tcn_{idx}_out")(x)
    x = layers.GlobalAveragePooling1D(name="tcn_gap")(x)
    return layers.Dense(int(latent_sequence.shape[-1]), activation="tanh", name="tcn_latent_proj")(x)
