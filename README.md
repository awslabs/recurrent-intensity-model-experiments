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

All algorithms require a testing set with labels. Most algorithms are trained from an autoregressive (self-supervised) training set without labels. Some algorithms are trained from (or in combination with) one or more validating set with labels.
The online matching setup uses the first validating set to infer the user-state distribution so that the CVX-Online algorithm can remain oblivious to the actual set of user (states) in the testing set for the purpose of online sumlations. (CVX-Online ignores the labels in that set.)

Here are the required fields of a supervised dataset for testing and validating purposes:

| attribute    | column name     | details                                                    |
|--------------|-----------------|------------------------------------------------------------|
| user_in_test | (index)         | <sub> indexed by USER_ID; allows duplicated indices w/ different TEST_START_TIME </sub> |
|              | TEST_START_TIME | to split between features and labels                       |
|              | `_hist_items`   | list of ITEM_IDs before TEST_START_TIME (exclusive)        |
|              | `_hist_ts`      | list of TIMESTAMPs before TEST_START_TIME (exclusive)      |
|              | `_hist_len`     | feature for user-popularity prior                          |
| item_in_test | (index)         | indexed by unique ITEM_ID                                  |
|              | `_hist_len`     | feature for item-popularity prior                          |
| target_csr   |                 | <sub> sparse matrix (user_in_test, item_in_test); sums up all events in testing horizon </sub> |
| horizon      | (default=inf)   | <sub> testing window after TEST_START_TIME for each user; agrees with target_csr </sub> |
| prior_score  | (default=None)  | <sub> sparse matrix (user_in_test, item_in_test) to allow exclude_train etc. </sub> |
| <sub> default_item_rec_top_k </sub>  | <sub> default=1% of item_in_test </sub> | <sub> default number of recs; further multiplied by mult variable in mtch experiments </sub> |
| <sub> default_user_rec_top_c </sub>  | <sub> default=1% of user_in_test </sub> | <sub> default number of recs; further multiplied by mult variable in mtch experiments </sub> |
| training_data |                | a reference to the autoregressive training set below |

Here are the subfields for an autoregressive (self-supervised) dataset for training purposes:

| attribute    | details                                                                    |
|--------------|----------------------------------------------------------------------------|
| user_df      | similar to user_in_test, but requires unique USER_ID (e.g., GroupBy.first) |
| item_df      | similar to item_in_test; count `_hist_len` by unique users                 |
| event_df     | agrees with the exploded `_hist_items` and `_hist_ts` from user_df         |


The testing, training, and validating sets can be conveniently created by `rime.dataset.base.create_dataset` or step-by-step illustrations in `rime.dataset.__init__.prepare_minimal_dataset`. The training set is bundled inside the testing set for convenience.

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

