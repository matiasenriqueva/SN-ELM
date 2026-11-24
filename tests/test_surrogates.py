import numpy as np
import pytest

from tests._original import original_archive, original_surrogate, requires_original
from zcatcmp.surrogates import ELM, SurrogateModel, TrainingSet


def _data(n=200, d=30, m=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, d))
    Y = np.column_stack([np.sin(X[:, :5]).sum(1), (X[:, 5:10] ** 2).sum(1)])[:, :m]
    return X, Y


@requires_original
@pytest.mark.parametrize("act", ["sigmoid", "multiquadric"])
@pytest.mark.parametrize("nh", [61, 241])        # N_H <= N_T y N_H > N_T
def test_elm_matches_original(act, nh):
    X, Y = _data()
    np.random.seed(7)
    old = original_surrogate.ELMSurrogate(nh, C=32.0, activation=act)
    old.fit(X, Y)
    np.random.seed(7)
    new = ELM(nh, C=32.0, activation=act)
    new.fit(X, Y)
    np.testing.assert_array_equal(new.predict(X[:50]), old.predict(X[:50]))


def _test_r2(model):
    Xt, Yt = _data(n=200, seed=1)
    resid = model.predict(Xt) - Yt
    return 1 - (resid ** 2).sum(0) / ((Yt - Yt.mean(0)) ** 2).sum(0)


@requires_original
def test_original_gaussian_is_degenerate():
    """Documenta el error corregido: anchos negativos -> H ~ 1e80 y R^2 < 0 fuera de la muestra."""
    X, Y = _data(n=600)
    np.random.seed(0)
    old = original_surrogate.ELMSurrogate(121, C=32.0, activation="gaussian")
    old.fit(X, Y)
    assert np.abs(old._H(X)).max() > 1e20
    assert (_test_r2(old) < 0).all()


def test_gaussian_is_well_conditioned():
    X, Y = _data(n=600)
    np.random.seed(0)
    elm = ELM(121, C=32.0, activation="gaussian")
    elm.fit(X, Y)
    H = elm._H(X)
    assert np.isfinite(H).all() and H.max() <= 1.0
    assert 0.01 < H.mean() < 0.99 and H.std() > 0.01
    np.random.seed(0)
    sigmoid = ELM(121, C=32.0, activation="sigmoid")
    sigmoid.fit(X, Y)
    r2, r2_sigmoid = _test_r2(elm), _test_r2(sigmoid)
    assert r2[0] > 0.9
    assert (r2 >= r2_sigmoid - 0.05).all()


@requires_original
def test_training_set_matches_original_archive():
    from jmetal.core.solution import FloatSolution

    rng = np.random.default_rng(3)
    X, F = rng.random((80, 4)), rng.random((80, 2))

    def sols(rows):
        out = []
        for x, f in rows:
            s = FloatSolution([0.0] * 4, [1.0] * 4, 2)
            s.variables, s.objectives = x.tolist(), f.tolist()
            out.append(s)
        return out

    old = original_archive.SurrogateArchive(30)
    old.add_initial(sols(zip(X[:30], F[:30])))
    old.update(sols(zip(X[30:], F[30:])))

    new = TrainingSet(30)
    new.add_initial(X[:30], F[:30])
    new.add(X[30:], F[30:])
    np.testing.assert_array_equal(new.X, np.array([s.variables for s in old.PT]))

    synced = TrainingSet(30)
    synced.sync(X[:50], F[:50])
    synced.sync(X, F)
    np.testing.assert_array_equal(synced.X, new.X)


@requires_original
def test_surrogate_model_normalization_matches_original():
    X, Y = _data(n=100)
    lb, ub = -np.ones(30) * 2, np.ones(30) * 2
    model = SurrogateModel(ELM(31, 1.0, "sigmoid"), lb, ub, 100)
    model.training.add_initial(X, Y)
    np.random.seed(5)
    model.fit()
    pred = model.predict(X[:10])
    # misma transformación que SurrogateArchive.training_arrays / denormalize_Y
    mu, sd = Y.mean(0), Y.std(0) + 1e-8
    Xn = 2.0 * (X - lb) / (ub - lb + 1e-8) - 1.0
    np.random.seed(5)
    ref = original_surrogate.ELMSurrogate(31, C=1.0, activation="sigmoid")
    ref.fit(Xn, (Y - mu) / sd)
    np.testing.assert_allclose(pred, ref.predict(Xn[:10]) * sd + mu, rtol=1e-12)


def test_scheduled_kriging_reuses_estimated_theta():
    from zcatcmp.surrogates import make_kriging

    X, Y = _data(n=120)
    krg = make_kriging(theta_every=3)
    thetas_seen = []
    for k in range(7):
        krg.fit(X[: 100 + k], Y[: 100 + k])
        thetas_seen.append([float(np.ravel(m.model.model["theta"])[0]) for m in krg.models])
    assert (krg.n_fits, krg.n_theta_fits) == (7, 3)              # ajustes 0, 3 y 6 estiman θ
    assert thetas_seen[1] == thetas_seen[2] == thetas_seen[0]    # entre medio, el θ estimado
    assert thetas_seen[4] == thetas_seen[5] == thetas_seen[3]

    # el reajuste usa el θ guardado, no el valor inicial del prototipo (1.0)
    krg.thetas = [0.123, 0.456]
    krg.fit(X[:110], Y[:110])                                    # ajuste 7: θ fijo
    assert [float(np.ravel(m.model.model["theta"])[0]) for m in krg.models] == [0.123, 0.456]


def test_frozen_theta_matches_optimized_model():
    from zcatcmp.surrogates import make_kriging

    X, Y = _data(n=150)
    full = make_kriging(theta_every=1)
    full.fit(X, Y)
    frozen = make_kriging(theta_every=2)
    frozen.fit(X, Y)
    frozen.fit(X, Y)                                             # segundo ajuste: θ fijo
    np.testing.assert_allclose(frozen.predict(X[:20]), full.predict(X[:20]), rtol=1e-10, atol=1e-10)
