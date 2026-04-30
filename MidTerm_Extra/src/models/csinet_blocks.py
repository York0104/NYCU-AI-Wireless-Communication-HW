from __future__ import annotations


def build_csinet_encoder(tf, encoded_dim: int, use_lightweight: bool = True, quant_bits: int = 8):
    layers = tf.keras.layers
    inputs = layers.Input(shape=(32, 32, 2), name="csinet_encoder_input")

    x = inputs
    if use_lightweight:
        x = layers.Conv2D(16, 3, padding="same", use_bias=False, name="stem_conv")(x)
        x = layers.BatchNormalization(name="stem_bn")(x)
        x = layers.LeakyReLU(name="stem_act")(x)
        x = layers.SeparableConv2D(32, 3, padding="same", use_bias=False, name="sep_conv")(x)
        x = layers.BatchNormalization(name="sep_bn")(x)
        x = layers.LeakyReLU(name="sep_act")(x)
        se = layers.GlobalAveragePooling2D(name="se_gap")(x)
        se = layers.Dense(8, activation="relu", name="se_fc1")(se)
        se = layers.Dense(32, activation="sigmoid", name="se_fc2")(se)
        se = layers.Reshape((1, 1, 32), name="se_reshape")(se)
        x = layers.Multiply(name="se_scale")([x, se])
        x = layers.Conv2D(8, 1, padding="same", activation="relu", name="proj_conv")(x)
    else:
        x = layers.Conv2D(2, 3, padding="same", name="baseline_conv")(x)
        x = layers.BatchNormalization(name="baseline_bn")(x)
        x = layers.LeakyReLU(name="baseline_act")(x)

    x = layers.Flatten(name="flatten")(x)
    latent = layers.Dense(encoded_dim, activation="tanh", name="latent_dense")(x)
    quantized = layers.Lambda(
        lambda t: tf.quantization.fake_quant_with_min_max_args(t, min=-1.0, max=1.0, num_bits=quant_bits),
        name=f"fake_quant_{quant_bits}bit",
    )(latent)
    return tf.keras.Model(inputs=inputs, outputs=quantized, name="csinet_encoder")


def build_csinet_decoder(tf, encoded_dim: int, residual_num: int = 2):
    layers = tf.keras.layers
    inputs = layers.Input(shape=(encoded_dim,), name="csinet_decoder_input")

    def common(y, name: str):
        y = layers.BatchNormalization(axis=-1, name=f"{name}_bn")(y)
        return layers.LeakyReLU(name=f"{name}_act")(y)

    def residual_block(y, index: int):
        shortcut = y
        y = layers.Conv2D(8, 3, padding="same", name=f"res{index}_conv1")(y)
        y = common(y, f"res{index}_c1")
        y = layers.Conv2D(16, 3, padding="same", name=f"res{index}_conv2")(y)
        y = common(y, f"res{index}_c2")
        y = layers.Conv2D(2, 3, padding="same", name=f"res{index}_conv3")(y)
        y = layers.BatchNormalization(axis=-1, name=f"res{index}_out_bn")(y)
        y = layers.Add(name=f"res{index}_add")([shortcut, y])
        return layers.LeakyReLU(name=f"res{index}_out_act")(y)

    x = layers.Dense(32 * 32 * 2, activation="linear", name="expand_dense")(inputs)
    x = layers.Reshape((32, 32, 2), name="decoder_reshape")(x)
    for idx in range(residual_num):
        x = residual_block(x, idx)
    outputs = layers.Conv2D(2, 3, padding="same", activation="sigmoid", name="decoder_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="csinet_decoder")
