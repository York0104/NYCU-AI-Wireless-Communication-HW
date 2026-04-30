from __future__ import annotations

from csinet_blocks import build_csinet_decoder, build_csinet_encoder


def build_lstm_baseline(
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
    temporal_latent = layers.LSTM(encoded_dim, name="temporal_lstm")(latent_seq)

    fused = layers.Concatenate(name="fused_concat")([current_latent, temporal_latent, doppler])
    fused = layers.Dense(encoded_dim, activation="tanh", name="fused_dense")(fused)
    output = decoder(fused)

    model = tf.keras.Model(inputs={"x_seq": x_seq, "doppler": doppler}, outputs=output, name="CsiNet_LSTM_Baseline")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
