import pytest, torch
import pandas as pd, numpy as np, scipy as sp


def test_rim_experiments_importable():
    import rim_experiments


@pytest.mark.parametrize("split_fn_name", ["split_by_time", "split_by_user"])
def test_synthetic_experiment(split_fn_name):
    from rim_experiments import main, plot_results
    self = main("prepare_synthetic_data", split_fn_name,
        lb_mult=[0.5, 0.2, 0.1, 0])
    fig = plot_results(self.results)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="skip in auto-tests")
@pytest.mark.parametrize("name", [
    "prepare_ml_1m_data",
    "prepare_netflix_data",
    "prepare_yoochoose_data",
])
def test_do_experiment(name):
    from rim_experiments import main
    main(name)


@pytest.mark.parametrize("maximization, expect", [
    (False, [[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
    (True, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
])
def test_solve_cvx(maximization, expect):
    """ we expect to see cvx = 3+4+3 < 1+4+9 = greedy """
    from rim_experiments.metrics.cvx import CVX
    score_mat = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    if not maximization:
        score_mat = 10-score_mat

    solver = CVX(score_mat, 1, 1) #, max_epochs=10, min_epsilon=0.001)
    pi = solver.fit(score_mat).transform(score_mat)
    v = solver.model.v.detach().numpy()[None, :]

    print(np.round(pi.toarray(), 2))
    print((pi * score_mat).sum())
    print(v)
    if expect is not None:
        assert np.allclose(pi.toarray(), expect, atol=0.1)
