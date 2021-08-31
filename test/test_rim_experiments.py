import pytest, torch
import pandas as pd, numpy as np, scipy as sp


def test_rim_experiments_importable():
    import rim_experiments


@pytest.mark.parametrize("split_fn_name", ["split_by_time", "split_by_user"])
def test_synthetic_experiment(split_fn_name):
    from rim_experiments import main, plot_results
    self = main("prepare_synthetic_data", split_fn_name,
        lb_mult=[0.5, 0.2, 0.1, 0])
    fig = plot_results(self)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="skip in auto-tests")
@pytest.mark.parametrize("name", [
    "prepare_ml_1m_data",
    "prepare_netflix_data",
    "prepare_yoochoose_data",
])
def test_do_experiment(name):
    from rim_experiments import main
    main(name)
