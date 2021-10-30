## Recurrent Intensity Model Experiments

![pytest workflow](https://github.com/awslabs/recurrent-intensity-model-experiments/actions/workflows/python-app.yml/badge.svg)

Repository to reproduce the experiments in the paper:

[Recurrent Intensity Modeling for User Recommendation and Online Matching](http://roseyu.com/time-series-workshop/submissions/2021/TSW-ICML2021_paper_47.pdf);
[(Another Link)](https://www.amazon.science/publications/recurrent-intensity-modeling-for-user-recommendation-and-online-matching)

```
@inproceedings{ma2021recurrent,
	Author = {Ma, Yifei and Liu, Ge and Deoras, Anoop},
	Booktitle = {ICML Time Series Workshop},
	Title = {Recurrent Intensity Modeling for User Recommendation and Online Matching},
	Year = {2021}
}
```

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
self = rime.Experiemnt(D, V, ...) # V is required only for Hawkes-Poisson and CVX-Online.
self.run()
self.results.print_results()
```

Here is what `Experiment.run` basically does:

**Step 1. Predictions.**

Let `x` be a user-time state and `y` be a unique item. Traditional top-k item-recommendation aims to predict `p(y|x)` for the next item given the user-state. However, we introduce symmetry via user-recommendation that allows for the comparisons across `x`. To this end, we novelly redefine the problem as the prediction of user-item engagement *intensities* in a unit time window in the immediate future, `λ(x,y)`, and utilize a marked temporal point process (MTPP) decomposition as `λ(x,y) = λ(x) p(y|x)`. Here is the code to do that:
```
rnn = rime.models.rnn.RNN(**self.model_hyps["RNN"]).fit(D.training_data)
hawkes = rime.models.hawkes.Hawkes(D.horizon).fit(D.training_data)
S = rnn.transform(D) * hawkes.transform(D)
```
S is a low-rank dataframe-like object with shape `(len(D.user_in_test), len(D.item_in_test))`.

**Step 2. Offline decisions.**

Ranking of the users (or items) and then comparing with the ground-truth targets can be laborsome. Instead, we utilize the `scipy.sparse` library to easily calculate the recommendation `hit` rates through point-wise multiplication. The sparsity property allows the evaluations to scale to large numbers of user/item pairs.
```
assigned_csr = rime.util._assign_topk(score_mat.T, C, device='cuda').T
metrics = rime.metrics.evaluate_assigned(df_to_coo(D.target_df), assigned_csr, …)
```

**Step 3. Online generalization.**

RIME contains an optional configuration *"CVX-Online"*, which simulates a scenario where we may not observe the full set of users ahead of time, but must make real-time decisions immediately and unregretfully as each user arrives one at a time.
This scenario is useful in the case of multi-day marketing campaigns with budgets allocated for the long-term prospects.
Our basic idea is to approximate a quantile threshold `v(y)` per item-y from an observable user sample and then generalize it to the testing set.
We pick the user sample from a "validation" data split `V`.
The item_in_test set must align between D and V because cvx also considers the competitions for limited user capacities from different items.
```
V = V.reindex(D.item_in_test.index, axis=1) # align on the item_in_test set to generalize
T = rnn.transform(V) * hawkes.transform(V)  # solve CVX based on the predicted scores.
cvx_online = rime.metrics.cvx.CVX(T.values, self._k1, self._c1, ...)
online_assignments = cvx_online.fit(T.values).transform(S.values)
out = rime.metrics.evaluate_assigned(df_to_coo(D.target_df), online_assignments, ...)
```

CVX-Online is integrated as `self.metrics_update("RNN-Hawkes", S, T)`,
when `self.online=True` and `T is not None`.

Finally, auto-generated documentation may be found at [ReadTheDocs](https://recurrent-intensity-model-experiments.readthedocs.io/).
To extend to other datasets, see example in [prepare_synthetic_data](src/rime/dataset/__init__.py).
The main functions are tested in [test](test).


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

