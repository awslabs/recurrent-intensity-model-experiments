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

4. Run matching experiment with CVX-Online allocation and plot diversity-relevance trade-off
   ```
   cvx_online = rime.Experiment(D, V, *V_extra,
                                mult=[0, 0.3, 0.7, 1, 3, 10, 200],  # turn on mtch calculation
                                cvx=True, online=True)  # optional; default => offline-greedy mtch
   cvx_online.run(["Rand", "Pop", "HP", "ALS", "BPR", "GraphConv-Extra",
                   "Transformer", "Transformer-Pop", "Transformer-HP",])
   cvx_online.print_results()  # tabular results
   fig = rime.util.plot_mtch_results(cvx_online)
   ```

    <img src="figure/online-ml-1m.png" alt="online-ml-1m" width="50%"/>

5. Run `pytest -s -x --pdb` for unit tests including the end-to-end workflow.

## Code Organization

**Step 0. Data Preparation**

The simplest way to prepare data is via `create_dataset` function:
```
rime.dataset.base.create_dataset(event_df: pd.DataFrame(columns=['USER_ID', 'ITEM_ID', 'TIMESTAMP']),
                                 user_df: pd.DataFrame(columns=['TEST_START_TIME'], index=USER_ID),
                                 item_df: pd.DataFrame(index=ITEM_ID),
                                 horizon: float >= 0 in the same unit as the TIMESTAMP column)
```
The `create_dataset` function will then regard events with `TIMESTAMP < TEST_START_TIME` as user histories and events within `TEST_START_TIME <= TIMESTAMP < TEST_START_TIME + horizon` as prediction targets. We collect user histories in dataframe order without artificial sorting by time. The collected user histories (`user_df._hist_items` and `user_df._hist_ts`) will be used for both auto-regressive training and user-side feature creation in prediction tasks. The function returns a `rime.dataset.base.Dataset` object.

To preserve temporal causality, we additionally filter users/items with at least 1 event as test candidates. The min-thresholds are adjustable and we record the filtered data as `user_in_test` and `item_in_test` attributes in the `Dataset` object. The filter does not apply for training, which is recorded separately as `user_df` and `item_df`.
We also filter by `user_in_test[TEST_START_TIME] < +inf`, which is an obvious necessity for test targets to exist and we abuse the +inf test-start time to indicate training-only users.
Another advanced case is having multiple test-start times for the same `USER_ID`, which we naturally handle by creating multiple rows in `user_in_test` with the corresponding histories. However, this case may also cause undesirable repetitions in the `user_df` attribute for training purposes. We thus deduplicate `user_df` by keeping the first entry per user in dataframe order. Similarly, we consider only the first row of each user in `user_df` towards the number of historical visits in `item_df._hist_len`.
For additional details, including a template for the created `Dataset` class as well as the use of `exclude_train` priors for cleaner model evaluations, please visit the example in `rime.dataset.__init__.prepare_minimal_dataset`.

For the `rime.Experiment` class to run, we need at least one dataset `D` for testing and auto-regressive training. We may optionally provide validating datasets `V` and `*V_extra` based on earlier time splits or user splits. The first validating dataset is used in the calibration of `CVX-Online` in Step 3 with the `online=True` option. All validating datasets are used by time-bucketed models (`GraphConv` and `HawkesPoisson`). Some models may be disabled if relevant data is missing.

**Step 1. Predictions**

Let `x` be a user-time state and `y` be a unique item. Traditional top-k item-recommendation aims to predict `p(y|x)` for the next item given the current user-state. On the other hand, we introduce symmetry via user-recommendation that allows for the comparisons across `x`. To this end, we novelly redefine the problem as the prediction of user-item engagement *intensities* in a unit time window in the immediate future, `λ(x,y)`, and utilize a marked temporal point process (MTPP) decomposition as `λ(x,y) = λ(x) p(y|x)`. Here is the code to do that:
```
rnn = rime.models.rnn.RNN(**self.model_hyps["RNN"]).fit(D.training_data)
hawkes = rime.models.hawkes.Hawkes(D.horizon).fit(D.training_data)
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

RIME contains an optional configuration *"CVX-Online"*, which simulates a scenario where we may not observe the full set of users ahead of time, but must make real-time decisions immediately and unregretfully as each user arrives one at a time.
This scenario is useful in the case of multi-day marketing campaigns with budgets allocated for the long-term prospects.
Our basic idea is to approximate a quantile threshold `v(y)` per item-y from an observable user sample and then generalize it to the testing set.
We pick the user sample from a "validation" data split `V`.
Additionally, we align the item_in_test between D and V, because cvx also considers the competitions for the limited user capacities from different items.
```
V = V.reindex(D.item_in_test.index, axis=1) # align on the item_in_test to generalize
T = rnn.transform(V) * hawkes.transform(V)  # solve CVX based on the validation set
cvx_online = rime.metrics.cvx.CVX(S, item_rec_topk, user_rec_C, ...) # set hyperparameters
online_assignments = cvx_online.fit(T).transform(S)
out = evaluate_assigned(D.target_csr, online_assignments, axis=0)
```

CVX-Online is integrated as `self.metrics_update("RNN-Hawkes", S, T)`,
when `self.online=True` and `T is not None`.

**Misc**

More information may be found in auto-generated documentation at [ReadTheDocs](https://recurrent-intensity-model-experiments.readthedocs.io/).
The main functions are covered in [test](test).


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

