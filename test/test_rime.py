import pytest, torch, tempfile
import numpy as np, scipy as sp
import scipy.sparse as sps


def test_rime_importable():
    import rime  # noqa: F401
    for name in dir(rime.models):
        obj = getattr(rime.models, name)
        if isinstance(obj, rime.util.MissingModel):
            raise ValueError(f"model {name} is not importable because {obj.err}")


def do_synthetic_common(*args, prepare_data_name="prepare_synthetic_data", **kw):
    from rime import main, plot_rec_results, plot_mtch_results
    self = main(prepare_data_name, *args, mult=[1.0], **kw)
    fig = plot_rec_results(self.results)  # noqa: F841
    fig2 = plot_mtch_results(self.results)  # noqa: F841

    with tempfile.NamedTemporaryFile("r") as fp:
        self.results.save_results(fp.name)
        print("saved results", fp.read())


def test_minimal_dataset():
    do_synthetic_common(prepare_data_name="prepare_minimal_dataset")


def test_minimal_dual():
    do_synthetic_common(prepare_data_name="prepare_minimal_dataset", dual=True, max_epochs=2)


@pytest.mark.parametrize("split_fn_name", ["split_by_time", "split_by_user"])
def test_synthetic_split(split_fn_name):
    do_synthetic_common(split_fn_name)


def test_synthetic_online():
    do_synthetic_common("split_by_user", online=True, max_epochs=2)


def test_synthetic_exclude_train():
    do_synthetic_common("split_by_user", True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="skip in auto-tests")
@pytest.mark.parametrize("name", [
    "prepare_ml_1m_data",
    "prepare_netflix_data",
    "prepare_yoochoose_data",
])
def test_do_experiment(name):
    from rime import main
    main(name)


@pytest.mark.parametrize("maximization, expect", [
    (False, [[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
    (True, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
])
def test_solve_dual(maximization, expect, **kw):
    """ we expect to see dual = 3+4+3 < 1+4+9 = greedy """
    from rime.metrics.dual import Dual
    score_mat = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    if not maximization:
        score_mat = 10 - score_mat

    solver = Dual(score_mat, alpha_ub=np.ones(3) / score_mat.shape[1],
                  beta_ub=np.ones(3) / score_mat.shape[0], **kw)
    pi = solver.fit(score_mat).transform(score_mat)
    pi = pi.numpy()
    v = solver.model.v.detach().numpy()[None, :]

    print(np.round(pi, 2))
    print((pi * score_mat).sum())
    print(v)
    if expect is not None:
        assert np.allclose(pi, expect, atol=0.1)


def test_score_array(shape=(3, 4), device="cpu"):
    from rime.util.score_array import LazyDenseMatrix, RandScore, score_op
    a = (LazyDenseMatrix(np.zeros((shape[0], 1))) @
         LazyDenseMatrix(np.zeros((shape[1], 1))).T).exp()
    b = sps.eye(*shape, 1)
    c = 3

    print(a.batch_size)

    score_op(a, "max")
    score_op(a + b, "max")
    score_op((a + b) * c, "max")
    score_op((a + b) * c, "max", device)
    score_op(a[[1, 2]], "max", device)
    score_op(a + RandScore.create(b.shape) * 2, "max", device)
    score_op(((a + b) * c + RandScore.create(b.shape) + 3)[[1, 2]], "max", device)
    print(((a + b) * c + RandScore.create(b.shape) + 3)[[1, 2]].traverse())
