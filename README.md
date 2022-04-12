## Recurrent Intensity Model Experiments

![pytest workflow](https://github.com/awslabs/recurrent-intensity-model-experiments/actions/workflows/python-app.yml/badge.svg)

Repository to reproduce the experiments in these papers:

[Bridging Recommendation and Marketing via Recurrent Intensity Modeling. ICLR 2022.](https://openreview.net/forum?id=TZeArecH2Nf)
```
@inproceedings{ma2022bridging,
    title={Bridging Recommendation and Marketing via Recurrent Intensity Modeling},
    author={Yifei Ma and Ge Liu and Anoop Deoras},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=TZeArecH2Nf}
}
```

[Recurrent Intensity Modeling for User Recommendation and Online Matching](http://roseyu.com/time-series-workshop/submissions/2021/TSW-ICML2021_paper_47.pdf);
[(Amazon Link)](https://www.amazon.science/publications/recurrent-intensity-modeling-for-user-recommendation-and-online-matching)

```
@inproceedings{ma2021recurrent,
    Author = {Ma, Yifei and Liu, Ge and Deoras, Anoop},
    Booktitle = {ICML Time Series Workshop},
    Title = {Recurrent Intensity Modeling for User Recommendation and Online Matching},
    Year = {2021}
}
```

## Getting Started

0. Manually install these packages as they may require a manual selection of cuda version
    - pytorch is pre-installed with aws deep-learning machine images (AMI)
    - `pip install dgl-cu111` # or matching cuda version with torch
    - `conda install -c conda-forge implicit implicit-proc=*=gpu -y`
    - If more dependencies are needed, try this: `conda env update --file environment.yml`
    - We use it to set up github build actions, with an obvious downside of not using any gpus.
1. Download and install via `pip install -e .`
2. Add data to the [data](data) folder. Some downloading and preparing scripts may be found in [data/util.py](data/util.py).
3. Run recommendation experiment as
    ```
    import rime
    D, V, *V_extra = rime.dataset.prepare_ml_1m_data(exclude_train=True)
    self = rime.Experiment(D, V, *V_extra)
    self.run()
    self.print_results()  # tabular results
    fig = rime.util.plot_rec_results(self, 'prec')
    ```

    <img src="figure/rec-ml-1m-prec.png" alt="rec-ml-1m-prec" width="40%"/>

    Notice the optional config that excludes training user-item pairs from reappearing in predictions (and targets) by automatically generating a prior_score attribute in dataset class. This helps non-temporal matrix-factorization models.

4. Run matching experiment with Dual-Online allocation and plot diversity-relevance trade-off
   ```
   online = rime.Experiment(D, V, *V_extra,
                            mult=[0, 0.3, 0.7, 1, 3, 10, 200],  # turn on match calculation
                            online=True)  # optional; default offline = greedy match
   online.run(["Rand", "Pop", "HP", "ALS", "BPR", "GraphConv-Extra",
                   "Transformer", "Transformer-Pop", "Transformer-HP",])
   online.print_results()  # tabular results
   fig = rime.util.plot_mtch_results(online)
   ```

    <img src="figure/online-ml-1m.png" alt="online-ml-1m" width="50%"/>

5. Run `pytest -s -x --pdb` for unit tests including the end-to-end workflow.

## Code Organization

**Step 0. Data Preparation**

The simplest way to prepare data is via `create_dataset` function:
```
rime.dataset.base.create_dataset(
    event_df: pd.DataFrame(columns=['USER_ID', 'ITEM_ID', 'TIMESTAMP']),
    user_df: pd.DataFrame(columns=['TEST_START_TIME'], index=USER_ID),
    item_df: pd.DataFrame(index=ITEM_ID),
    horizon: float >= 0 in the same unit as the TIMESTAMP column)
```
Inputs to this function include event_df for both training and holdout data, user_df for all users, and item_df for all items. The holdout test window is constructed per user to be between `TEST_START_TIME <= TIMESTAMP < TEST_START_TIME + horizon`, where `TEST_START_TIME` is a required column in user_df. The user_df may contain repeated user rows with different `TEST_START_TIME`, in which case they will be treated as different `test_requests`. Also in the case of repeated user rows, the first occurance of the same user in the original unsorted order is used to decide for their auto-regressive training data. The function returns a `rime.dataset.base.Dataset` object, where additional feature aggregation is automatically conducted for convenience.

As discussed in the paper, `create_dataset` has a default option to filter `min_user_len=min_item_len>=1` for unbiased evaluation. These thresholds can be set to zeros if cold-start is considered. We also filter by `TEST_START_TIME<inf`, without which the test window would not exist. See `rime.dataset.__init__.prepare_minimal_dataset` for some examples including these special cases.

For the `rime.Experiment` class to run, we need at least one dataset `D` for testing and auto-regressive training. We may optionally provide validating datasets `V` and `*V_extra` based on earlier time splits or user splits. The first validating dataset is used in the calibration of `Dual-Online` in Step 3 with the `online=True` option. All validating datasets are used by time-bucketed models (`GraphConv` and `HawkesPoisson`). Some models may be disabled if relevant data is missing.

**Step 1. Predictions**

Let `x` be a user-time state and `y` be a unique item. Traditional top-k item-recommendation aims to predict `p(y|x)` for the next item given the current user-state. On the other hand, we introduce symmetry via user-recommendation that allows for the comparisons across `x`. To this end, we novelly redefine the problem as the prediction of user-item engagement *intensities* in a unit time window in the immediate future, `λ(x,y)`, and utilize a marked temporal point process (MTPP) decomposition as `λ(x,y) = λ(x) p(y|x)`. Here is the code to do that:
```
rnn = rime.models.rnn.RNN(**self.model_hyps["RNN"]).fit(D.auto_regressive)
hawkes = rime.models.hawkes.Hawkes(D.horizon).fit(D.auto_regressive)
S = rnn.transform(D) * hawkes.transform(D)
```
S is a low-rank dataframe-like object with shape `(len(D.user_in_test), len(D.item_in_test))`.

**Step 2. Offline decisions**

Ranking of the items (or users) and then comparing with the ground-truth targets can be laborsome. Instead, we utilize the `scipy.sparse` library to easily calculate the recommendation `hit` rates through point-wise multiplication. The sparsity property allows the evaluations to scale to large numbers of user-item pairs.
```
item_rec_assignments = rime.util._assign_topk(S, item_rec_topk, device='cuda')
item_rec_metrics = evaluate_assigned(D.target_csr, item_rec_assignments, axis=1, device='cuda')
user_rec_assignments = rime.util._assign_topk(S.T, user_rec_C, device='cuda').T
user_rec_metrics = evaluate_assigned(D.target_csr, user_rec_assignments, axis=0, device='cuda')
```

**Step 3. Online simulation**

RIME contains an optional configuration *"Dual-Online"*, which simulates a scenario where we may not observe the full set of users ahead of time, but must make real-time decisions immediately and unregretfully as each user arrives one at a time.
This scenario is useful in the case of multi-day marketing campaigns with budgets allocated for the long-term prospects.
Our basic idea is to approximate a quantile threshold `v(y)` per item-y from an observable user sample and then generalize it to the testing set.
We pick the user sample from a "validation" data split `V`.
Additionally, we align the item_in_test between D and V, because Dual also considers the competitions for the limited user capacities from different items.
```
V = V.reindex(D.item_in_test.index, axis=1) # align on the item_in_test to generalize
T = rnn.transform(V) * hawkes.transform(V)  # solve Dual based on the validation set
dual = rime.metrics.dual.Dual(S, item_rec_topk, user_rec_C, ...) # set hyperparameters
dual_assigned = dual.fit(T).transform(S)
out = evaluate_assigned(D.target_csr, dual_assigned, axis=0)
```

Dual-Online is integrated as `self.metrics_update("RNN-Hawkes", S, T)`,
when `self.online=True` and `T is not None`.

**Misc**

More information may be found in auto-generated documentation at [ReadTheDocs](https://recurrent-intensity-model-experiments.readthedocs.io/).
The main functions are covered in [test](test).


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

