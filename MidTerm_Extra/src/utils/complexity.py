from __future__ import annotations

import time


def count_trainable_params(model) -> int:
    return int(sum(int(v.shape.num_elements()) for v in model.trainable_weights))


def average_inference_seconds(model, inputs, repeats: int = 3) -> float:
    elapsed = []
    for _ in range(repeats):
        started = time.time()
        model.predict(inputs, verbose=0)
        elapsed.append(time.time() - started)
    batch_size = len(next(iter(inputs.values())))
    return float(sum(elapsed) / len(elapsed) / max(1, batch_size))
