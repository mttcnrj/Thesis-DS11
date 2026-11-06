"""Required test 7: same seed + config -> identical predictions (atol 1e-6), determinism on (D15)."""
import numpy as np
import pytest

pytestmark = pytest.mark.fast


def test_same_seed_same_predictions():
    tf = pytest.importorskip("tensorflow")
    from pasig.models.hybrid import HybridConfig, build_hybrid
    from pasig.train.loop import set_seed
    tf.config.set_visible_devices([], "GPU")
    rng = np.random.default_rng(0)
    Xs = rng.random((256, 24, 3)).astype("float32")
    Xt = rng.random((256, 10)).astype("float32")
    Y = rng.random((256, 6)).astype("float32")
    Y[::7, 2] = -1.0                              # invalid targets are masked
    preds = []
    for _ in range(2):
        set_seed(7, deterministic=True)
        m = build_hybrid(HybridConfig(lstm_units=16, gru_units=8, dense_units=8, head_units=8, batch=64), (24, 3), 10)
        m.fit({"seq": Xs, "tab": Xt}, Y, epochs=1, batch_size=64, verbose=0)
        preds.append(m.predict({"seq": Xs[:32], "tab": Xt[:32]}, verbose=0))
    assert np.allclose(preds[0], preds[1], atol=1e-6)


def test_add_back_layer_returns_level_plus_change():
    pytest.importorskip("tensorflow")
    from pasig.models.hybrid import HybridConfig, build_hybrid
    m = build_hybrid(HybridConfig(lstm_units=8, gru_units=4, dense_units=4, head_units=4), (24, 2), 3)
    for w in m.get_layer("change").weights:      # zero change -> output = scaled level at t for every lead
        w.assign(np.zeros(w.shape, dtype="float32"))
    Xs = np.random.default_rng(1).random((5, 24, 2)).astype("float32")
    p = m.predict({"seq": Xs, "tab": np.zeros((5, 3), "float32")}, verbose=0)
    assert np.allclose(p, np.repeat(Xs[:, -1, 0:1], 6, axis=1), atol=1e-6)


def test_masked_mse_ignores_invalid_targets():
    pytest.importorskip("tensorflow")
    from pasig.models.hybrid import masked_mse, weighted_masked_mse
    yt = np.array([[0.5, -1.0], [0.2, 0.9]], "float32")
    yp = np.array([[0.4, 100.0], [0.2, 0.7]], "float32")
    assert np.isclose(float(masked_mse(yt, yp)), (0.01 + 0 + 0.04) / 3, atol=1e-6)
    w = weighted_masked_mse(5.0, 0.8)          # target 0.9 >= 0.8 gets weight 5
    assert np.isclose(float(w(yt, yp)), (0.01 + 0 + 5 * 0.04) / 3, atol=1e-6)
