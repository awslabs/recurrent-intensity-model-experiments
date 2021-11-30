import pytest, torch, tempfile
import pandas as pd, numpy as np, scipy as sp
import scipy.sparse as sps


def test_rime_importable():
    import rime


def do_synthetic_common(*args, prepare_data_name="prepare_synthetic_data", **kw):
    from rime import main, plot_results
    self = main(prepare_data_name, *args, mult=[1.0], **kw)
    fig = plot_results(self.results)

    with tempfile.NamedTemporaryFile("r") as fp:
        self.results.save_results(fp.name)
        print("saved results", fp.read())


def test_minimal_dataset():
    do_synthetic_common(prepare_data_name="prepare_minimal_dataset")


def test_minimal_cvx():
    do_synthetic_common(prepare_data_name="prepare_minimal_dataset", cvx=True, max_epochs=2)


@pytest.mark.parametrize("split_fn_name", ["split_by_time", "split_by_user"])
def test_synthetic_split(split_fn_name):
    do_synthetic_common(split_fn_name)


@pytest.mark.parametrize("online", [False, True])
def test_synthetic_cvx_online(online):
    do_synthetic_common("split_by_user", cvx=True, online=online, max_epochs=2)


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
def test_solve_cvx(maximization, expect, **kw):
    """ we expect to see cvx = 3+4+3 < 1+4+9 = greedy """
    from rime.metrics.cvx import CVX
    score_mat = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    if not maximization:
        score_mat = 10-score_mat

    solver = CVX(score_mat, 1, 1, **kw)
    pi = solver.fit(score_mat).transform(score_mat)
    if sp.sparse.issparse(pi):
        pi = pi.toarray()
    v = solver.model.v.detach().numpy()[None, :]

    print(np.round(pi, 2))
    print((pi * score_mat).sum())
    print(v)
    if expect is not None:
        assert np.allclose(pi, expect, atol=0.1)


def test_score_array(shape=(3, 4), device="cpu"):
    from rime.util.score_array import LowRankDataFrame, RandScore, score_op
    index = np.arange(shape[0]) + 2
    columns = np.arange(shape[1]) + 5

    a = LowRankDataFrame(
        np.zeros((len(index), 2)), np.zeros((len(columns), 2)), index, columns, 'exp'
        )
    b = sps.eye(len(index), len(columns), 1)
    c = 3

    print(a.batch_size)

    score_op(a, "max")
    score_op(a + b, "max")
    score_op((a + b) * c, "max")
    score_op((a + b) * c, "max", device)
    score_op(a.reindex(np.asarray([3, 4, 5]), axis=0, fill_value=0), "max", device)
    score_op(a + RandScore.like(b) * 2, "max", device)
