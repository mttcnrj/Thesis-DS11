"""Hybrid LSTM-GRU with a dense tabular path (Figure III-4) and the standalone LSTM.

Moved from scripts/smoke/common.py::build_hybrid / build_lstm. Changes for Keras 3 serialisation only:
the `Lambda(lambda s: s[:, -1, 0:1])` that takes the scaled Pandacan level at t is a registered layer
(`LevelAtT`), and the losses are registered functions, so saved checkpoints reload in Notebook 06.
Keep the cuDNN path: default activations, recurrent_dropout = 0, unroll = False, no masking layers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import keras
from keras import layers, ops

LEADS = 6


@dataclass
class HybridConfig:
    lstm_layers: int = 1
    lstm_units: int = 64
    gru_units: int = 32
    dense_layers: int = 1
    dense_units: int = 16
    head_units: int = 32
    dropout: float = 0.2
    lr: float = 1e-3
    batch: int = 128

    def tag(self) -> str:
        return (f"L{self.lstm_layers}x{self.lstm_units}-G{self.gru_units}-D{self.dense_layers}x{self.dense_units}"
                f"-do{self.dropout}-lr{self.lr}-b{self.batch}")

    def as_dict(self) -> dict:
        return asdict(self)


@keras.saving.register_keras_serializable(package="pasig")
class LevelAtT(layers.Layer):
    """Scaled Pandacan level at t = sequence channel 0 at the last step (D9 add-back)."""

    def call(self, s):
        return s[:, -1, 0:1]


@keras.saving.register_keras_serializable(package="pasig")
def masked_mse(y_true, y_pred):
    """MSE over valid targets only; invalid targets are encoded as y_true = -1."""
    mask = ops.cast(y_true > -0.5, y_pred.dtype)
    se = ops.square(y_true - y_pred) * mask
    return ops.sum(se) / ops.maximum(ops.sum(mask), 1.0)


def weighted_masked_mse(weight: float, threshold_scaled: float):
    """D20: masked MSE with weight w on valid targets >= the high-water level (scaled), 1 otherwise.
    The weight uses the true future level, allowed in a training loss but never as an input."""

    @keras.saving.register_keras_serializable(package="pasig", name=f"wmse_{weight:g}_{threshold_scaled:.6f}")
    def loss(y_true, y_pred):
        mask = ops.cast(y_true > -0.5, y_pred.dtype)
        w = ops.where(y_true >= threshold_scaled, weight, 1.0) * mask
        return ops.sum(ops.square(y_true - y_pred) * w) / ops.maximum(ops.sum(mask), 1.0)

    return loss


def _head(seq_in, x, add_back: bool):
    if add_back:   # D9: predict the change from the current level; a fixed layer adds L_t back
        delta = layers.Dense(LEADS, activation="linear", name="change")(x)
        level_t = LevelAtT(name="level_t")(seq_in)
        return layers.Add(name="levels")([delta, layers.Concatenate()([level_t] * LEADS)])
    return layers.Dense(LEADS, activation="linear", name="levels")(x)


def build_hybrid(cfg: HybridConfig, seq_shape, n_tab: int, use_tabular: bool = True,
                 second_lstm_instead_of_gru: bool = False, add_back: bool = True, loss=masked_mse):
    seq_in = keras.Input(shape=seq_shape, name="seq")
    x = seq_in
    for _ in range(cfg.lstm_layers):
        x = layers.LSTM(cfg.lstm_units, return_sequences=True)(x)
        x = layers.Dropout(cfg.dropout)(x)            # dropout BETWEEN layers; never recurrent_dropout
    x = layers.LSTM(cfg.gru_units)(x) if second_lstm_instead_of_gru else layers.GRU(cfg.gru_units)(x)
    x = layers.Dropout(cfg.dropout)(x)
    inputs = [seq_in]
    if use_tabular:
        tab_in = keras.Input(shape=(n_tab,), name="tab")
        z = tab_in
        for _ in range(cfg.dense_layers):
            z = layers.Dense(cfg.dense_units, activation="relu")(z)
        x = layers.Concatenate()([x, z])
        inputs.append(tab_in)
    x = layers.Dense(cfg.head_units, activation="relu")(x)
    model = keras.Model(inputs, _head(seq_in, x, add_back))
    model.compile(optimizer=keras.optimizers.Adam(cfg.lr), loss=loss)
    return model


def build_lstm(units: int, dropout: float, lr: float, seq_shape, add_back: bool = True, loss=masked_mse):
    """Standalone LSTM: the tabular vector is repeated at every time step upstream (data.repeat_tab)."""
    seq_in = keras.Input(shape=seq_shape, name="seq")
    x = layers.LSTM(units)(seq_in)
    x = layers.Dropout(dropout)(x)
    model = keras.Model(seq_in, _head(seq_in, x, add_back))
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss)
    return model
