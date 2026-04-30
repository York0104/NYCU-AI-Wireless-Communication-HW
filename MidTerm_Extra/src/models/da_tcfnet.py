from __future__ import annotations

from csinet_blocks import build_csinet_decoder, build_csinet_encoder
from tcn_block import build_tcn_feature


def build_da_tcfnet(
    tf,
    time_steps: int,
    encoded_dim: int = 512,
    residual_num: int = 2,
    quant_bits: int = 8,
    freeze_encoder: bool = False,
):
    layers = tf.keras.layers

    x_seq = layers.Input(shape=(time_steps, 32, 32, 2), name="x_seq")
    doppler = layers.Input(shape=(1,), name="doppler")

    encoder = build_csinet_encoder(tf, encoded_dim=encoded_dim, use_lightweight=True, quant_bits=quant_bits)
    decoder = build_csinet_decoder(tf, encoded_dim=encoded_dim, residual_num=residual_num)
    encoder.trainable = not freeze_encoder

    latent_seq = layers.TimeDistributed(encoder, name="time_distributed_encoder")(x_seq)
    current_latent = layers.Lambda(lambda t: t[:, -1, :], name="current_latent")(latent_seq)
    temporal_latent = build_tcn_feature(tf, latent_seq, filters=min(256, max(64, encoded_dim // 2)))

    gate_input = layers.Concatenate(name="gate_input")([current_latent, temporal_latent, doppler])
    gate_hidden = layers.Dense(128, activation="relu", name="gate_fc1")(gate_input)
    alpha = layers.Dense(encoded_dim, activation="sigmoid", name="gate_alpha")(gate_hidden)

    weighted_current = layers.Multiply(name="weighted_current")([alpha, current_latent])
    one_minus_alpha = layers.Lambda(lambda a: 1.0 - a, name="one_minus_alpha")(alpha)
    weighted_temporal = layers.Multiply(name="weighted_temporal")([one_minus_alpha, temporal_latent])
    fused_latent = layers.Add(name="fused_latent")([weighted_current, weighted_temporal])

    coarse = decoder(current_latent)
    refined = decoder(fused_latent)
    fusion_map = layers.Concatenate(name="recon_concat")([coarse, refined])
    fusion_map = layers.Conv2D(8, 3, padding="same", activation="relu", name="refine_conv1")(fusion_map)
    output = layers.Conv2D(2, 1, padding="same", activation="sigmoid", name="reconstruction")(fusion_map)

    model = tf.keras.Model(inputs={"x_seq": x_seq, "doppler": doppler}, outputs=output, name="DA_TCFNet")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
