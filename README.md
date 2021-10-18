## Recurrent Intensity Model Experiments

![pytest workflow](https://github.com/awslabs/recurrent-intensity-model-experiments/actions/workflows/python-app.yml/badge.svg)

Repository to reproduce the experiments in the paper:

Yifei Ma, Ge Liu Anoop Deoras. Recurrent Intensity Modeling for User Recommendation and Online Matching. ICML 2021 Time-Series Workshop.
[link1](http://roseyu.com/time-series-workshop/submissions/2021/TSW-ICML2021_paper_47.pdf)
[link2](https://www.amazon.science/publications/recurrent-intensity-modeling-for-user-recommendation-and-online-matching)


## Getting Started

1. Download and install via `pip install -e .` Additional conda dependencies may be found at [environment.yml](environment.yml)
2. Add data to the [data](data) folder. Some downloading and preparing scripts may be found in [data/util.py](data/util.py).
3. Run experiment as
    ```
    from rime import main, plot_results
    mult=[0, 0.1, 0.2, 0.5, 1, 3, 10, 30, 100]
    self = main("prepare_ml_1m_data", mult=mult)
    fig = plot_results(self)
    ```
    ![greedy-ml-1m](figure/greedy-ml-1m.png)
    ```
    cvx_online = main("prepare_ml_1m_data", mult=mult, cvx=True, online=True)
    fig = plot_results(cvx_online)
    ```
    ![online-ml-1m](figure/online-ml-1m.png)

## Code Organization

Here is the content of the `main` function:
```
D, V = prepare_some_dataset(...) # output instances of rime.dataset.base.Dataset
self = rime.Experiemnt(D, V, ...)
self.run()
self.results.print_results()
```

Here is what `Experiment.run` basically does:
```
rnn = rime.models.rnn.RNN(**self.model_hyps["RNN"]).fit(D.training_data)
hawkes = rime.models.hawkes.Hawkes(D.horizon).fit(D.training_data)
S = rnn.transform(D) * hawkes.transform(D)  # output shape (D.user_in_test, D.item_in_test)
self.metrics_update("RNN", S)
```

OnlnMtch does not allow leakage of `D.user_in_test`. Instead, CVX is calibrated by `V.user_in_test`:
```
T = rnn.transform(V) * hawkes.transform(V)
T = T.reindex(D.item_in_test.index, axis=1, fill_value=0)
                                            # output shape (V.user_in_test, D.item_in_test)
cvx = rime.metrics.cvx.CVX(T.values, self._k1, self._c1, ...)
online_assignments = cvx.fit(T.values).transform(S.values)
out = rime.metrics.evaluate_assigned(df_to_coo(D.target_df), online_assignments, ...)
```

OnlnMtch is integrated as `self.metrics_update("RNN", S, T)`,
when `T is not None` and `self.online=True`.

Auto-generated documentation may be found at [ReadTheDocs](https://recurrent-intensity-model-experiments.readthedocs.io/).
To extend to other datasets, see example in [prepare_synthetic_data](src/rime/dataset/__init__.py).
The main functions are tested in [test](test).


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

