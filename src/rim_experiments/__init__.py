import functools, collections, torch, dataclasses
from typing import Dict, List
from rim_experiments.models import *
from rim_experiments.metrics import *
from rim_experiments.dataset import Dataset
from rim_experiments import dataset
from rim_experiments.util import _argsort, cached_property, df_to_coo


@dataclasses.dataclass
class ExperimentResult:
    _k1: int
    _c1: int
    _kmax: int
    _cmax: int
    item_ppl: float
    user_ppl: float

    item_rec: Dict[str, Dict[str, float]] = dataclasses.field(default_factory=dict)
    user_rec: Dict[str, Dict[str, float]] = dataclasses.field(default_factory=dict)
    ub_: Dict[str, List[Dict[str, float]]] = dataclasses.field(default_factory=dict)
    lb_: Dict[str, List[Dict[str, float]]] = dataclasses.field(default_factory=dict)

    def print_results(self):
        print('\nitem_rec')
        print(pd.DataFrame(self.item_rec).T)
        print('\nuser_rec')
        print(pd.DataFrame(self.user_rec).T)

    def get_mtch_(self, k=None, c=None, name="ub_"):
        y = {}
        for method, x in getattr(self, name).items():
            x = pd.DataFrame(x)
            if k is not None:
                y[method] = x.set_index('k').loc[k].set_index('c').sort_index().T
            else:
                y[method] = x.set_index('c').loc[c].set_index('k').sort_index().T
        return pd.concat(y, axis=1) if len(y) else None


class Experiment:
    def __init__(self, D, V,
        ub_mult=[], # [3, 10, 30, 100], # sweep for constraints to generate curve
        lb_mult=[], # [0.5, 0.2, 0.1, 0],
        models_to_run=[
            "Rand", "Pop", "EMA", "Hawkes", "HP",
            "RNN", "RNN-Pop", "RNN-EMA", "RNN-Hawkes", "RNN-HP",
            "BPR-Item", "BPR-User"
            ],
        model_hyps={},
        device="cpu",
        ):
        self.D = D
        self.V = V

        self.ub_mult = np.array(ub_mult)
        self.lb_mult = np.array(lb_mult)
        self.models_to_run = models_to_run
        self.model_hyps = model_hyps
        self.device = device

        self.results = ExperimentResult(
            _k1 = self.D.default_item_rec_top_k,
            _c1 = self.D.default_user_rec_top_c,
            _kmax = len(self.D.item_in_test),
            _cmax = len(self.D.user_in_test),
            item_ppl = self.D.get_stats()['event_df']['item_ppl'],
            user_ppl = self.D.get_stats()['event_df']['user_ppl'],
        )

        # pass-through references
        self.__dict__.update(self.results.__dict__)
        self.print_results = self.results.print_results
        self.get_mtch_ = self.results.get_mtch_


    def metrics_update(self, name, S):
        target_csr = df_to_coo(self.D.target_df)
        score_mat = self.D.transform(S).values

        self.item_rec[name] = evaluate_item_rec(target_csr, score_mat, self._k1)
        self.user_rec[name] = evaluate_user_rec(target_csr, score_mat, self._c1)

        print(pd.DataFrame({k:v for k,v in {
            'item_rec': self.item_rec[name],
            'user_rec': self.user_rec[name],
        }.items() if v is not None}).T)

        if len(self.ub_mult):
            self.ub_[name] = self._mtch_update(target_csr, score_mat, "ub", self.ub_mult)

        if len(self.lb_mult):
            self.lb_[name] = self._mtch_update(target_csr, score_mat, "lb", self.lb_mult)


    def _mtch_update(self, target_csr, score_mat, constraint_type, mult):
        confs = ([(self._k1, self._c1, "ub")] # equality constraint
            + [(k, self._c1, constraint_type) for k in (self._k1 * mult)]
            + [(self._k1, c, constraint_type) for c in (self._c1 * mult)]
        )

        argsort_ij = _argsort(score_mat, device=self.device)

        out = []
        for k, c, constraint_type in confs:
            res = evaluate_mtch(
                target_csr, score_mat, k, c, argsort_ij, constraint_type
            )
            res.update({'k': k, 'c': c})
            out.append(res)

        return out


    def _check_run_model(self, name, S_fn):
        if name in self.models_to_run:
            print("running", name)
            self.metrics_update(name, S_fn())


    def run(self):
        self._check_run_model("Rand",
            lambda: Rand().transform(self.D))

        self._check_run_model("Pop",
            lambda: Pop().transform(self.D))

        self._check_run_model("EMA",
            lambda: Pop(user_rec=False, item_rec=True).transform(self.D) *
                EMA(self.D.horizon).transform(self.D))

        self._check_run_model("Hawkes",
            lambda: Pop(user_rec=False, item_rec=True).transform(self.D) *
                self._hawkes.transform(self.D))

        self._check_run_model("HP",
            lambda: Pop(user_rec=False, item_rec=True).transform(self.D) *
                self._hawkes_poisson.transform(self.D))

        self._check_run_model("RNN",
            lambda: self._rnn.transform(self.D))

        self._check_run_model("RNN-Pop",
            lambda: self._rnn.transform(self.D) *
                Pop(user_rec=True, item_rec=False).transform(self.D))

        self._check_run_model("RNN-EMA",
            lambda: self._rnn.transform(self.D) * EMA(self.D.horizon).transform(self.D))

        self._check_run_model("RNN-Hawkes",
            lambda: self._rnn.transform(self.D) * self._hawkes.transform(self.D))

        self._check_run_model("RNN-HP",
            lambda: self._rnn.transform(self.D) * self._hawkes_poisson.transform(self.D))

        self._check_run_model("BPR-Item",
            lambda: LightFM_BPR(item_rec=True).fit(self.D).transform(self.D))

        self._check_run_model("BPR-User",
            lambda: LightFM_BPR(user_rec=True).fit(self.D).transform(self.D))


    @cached_property
    def _rnn(self):
        fitted = RNN(self.D.item_df, **self.model_hyps.get("RNN", {})).fit(self.D)
        for name, param in fitted.model.named_parameters():
            print(name, param.data.shape)
        return fitted

    @cached_property
    def _hawkes(self):
        return Hawkes(self.D.horizon).fit(self.D)

    @cached_property
    def _hawkes_poisson(self):
        return HawkesPoisson(self._hawkes).fit(self.V)


def main(name, *args, **kw):
    prepare_fn = getattr(dataset, name)
    D, V = prepare_fn(*args)
    self = Experiment(D, V, **kw)
    self.run()
    self.results.print_results()
    return self


def plot_results(self, plot_lb=True, plot_ub=True, logy=True):
    """ self is an instance of Experiment or ExperimentResult """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))
    df = [self.get_mtch_(k=self._k1), self.get_mtch_(c=self._c1)]
    lb = [self.get_mtch_(k=self._k1, name="lb_"),
          self.get_mtch_(c=self._c1, name="lb_")]

    xname = [f'ItemRec Prec@{self._k1}', f'UserRec Prec@{self._c1}']
    yname = ['item_ppl', 'user_ppl']

    for ax, df, lb, xname, yname in zip(ax, df, lb, xname, yname):
        if plot_lb and lb is not None:
            ax.plot(
                lb.loc['prec'].unstack().values.T,
                lb.loc[yname].unstack().values.T,
                '+-',
            )
            ax.set_prop_cycle(None)
        if plot_ub and df is not None:
            ax.plot(
                df.loc['prec'].unstack().values.T,
                df.loc[yname].unstack().values.T,
                '+:',
            )
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        if logy:
            ax.set_yscale('log')
    fig.legend(
        (df if df is not None else lb).loc['prec'].unstack().index.values,
        bbox_to_anchor=(0.1, 0.9, 0.8, 0), loc=3, ncol=4,
        mode="expand", borderaxespad=0.)
    fig.subplots_adjust(wspace=0.25)
    return fig
