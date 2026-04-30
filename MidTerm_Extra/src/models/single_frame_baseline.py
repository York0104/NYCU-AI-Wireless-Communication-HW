from __future__ import annotations

from csinet_blocks import build_csinet_decoder, build_csinet_encoder


def build_single_frame_baseline(
    tf,
    time_steps: int,
    encoded_dim: int = 512,
    residual_num: int = 2,
    quant_bits: int = 8,
):
    layers = tf.keras.layers

    x_seq = layers.Input(shape=(time_steps, 32, 32, 2), name="x_seq")
    doppler = layers.Input(shape=(1,), name="doppler")

    encoder = build_csinet_encoder(tf, encoded_dim=encoded_dim, use_lightweight=True, quant_bits=quant_bits)
    decoder = build_csinet_decoder(tf, encoded_dim=encoded_dim, residual_num=residual_num)

    current_frame = layers.Lambda(lambda t: t[:, -1, :, :, :], name="current_frame")(x_seq)
    current_latent = encoder(current_frame)
    output = decoder(current_latent)

    model = tf.keras.Model(
        inputs={"x_seq": x_seq, "doppler": doppler},
        outputs=output,
        name="CsiNet_Single_Frame_Baseline",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
