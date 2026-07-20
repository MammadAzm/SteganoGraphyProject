from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_global_seed(
    seed: int,
    *,
    deterministic: bool = True,
) -> dict[str, Any]:
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer.")

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ.setdefault(
            "TF_DETERMINISTIC_OPS",
            "1",
        )

    random.seed(seed)
    np.random.seed(seed)

    status: dict[str, Any] = {
        "python": True,
        "numpy": True,
        "tensorflow": False,
        "seed": seed,
        "deterministic": deterministic,
    }

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)

        try:
            tf.keras.utils.set_random_seed(seed)
        except AttributeError:
            pass

        if deterministic:
            try:
                tf.config.experimental.enable_op_determinism()
            except (AttributeError, RuntimeError):
                pass

        status["tensorflow"] = True

    except ImportError:
        status["tensorflow"] = False

    return status