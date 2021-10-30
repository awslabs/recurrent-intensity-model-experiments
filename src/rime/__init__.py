try:
    import matplotlib.pyplot as plt
    plt.plot(); plt.show() # tick behaves weirdly with matplotlib
except ImportError:
    pass

import functools, collections, torch, dataclasses, warnings, json
from typing import Dict, List
from rime.models import *
from rime.metrics import *
from rime import dataset
from rime.util import _argsort, cached_property, df_to_coo


@dataclasses.dataclass
class ExperimentResult:
    cvx: bool
    online: bool
    _k1: int
    _c1: int
    _kmax: int
    _cmax: int
    item_ppl: float
    user_ppl: float

    item_rec: Dict[str, Dict[str, float]] = dataclasses.field(default_factory=dict)
    user_rec: Dict[str, Dict[str, float]] = dataclasses.field(default_factory=dict)
    mtch_: Dict[str, List[Dict[str, float]]] = dataclasses.field(default_factory=dict)

    def print_results(self):
        print('\nitem_rec')
        print(pd.DataFrame(self.item_rec).T)
        print('\nuser_rec')
        print(pd.DataFrame(self.user_rec).T)

        mtch1 = self.get_mtch_(self._k1, self._c1)
        if mtch1 is not None:
            print('\nmtch_')
            print(mtch1.T)

    def save_results(self, fn):
        with open(fn, 'w') as fp:
            json.dump(dataclasses.asdict(self), fp)

    def get_mtch_(self, k=None, c=None, name="mtch_"):
        y = {}
        for method, x in getattr(self, name).items():
            x = pd.DataFrame(x)
            if k is not None and c is not None:
                y[method] = x.set_index(['k', 'c']).loc[(k, c)]
            elif k is not None:
                y[method] = x.set_index(['k', 'c']).loc[k].sort_index().T
            elif c is not None:
                y[method] = x.set_index(['c', 'k']).loc[c].sort_index().T
            else:
                raise ValueError("either k or c must be provided")
        return pd.concat(y, axis=1) if len(y) else None


class Experiment:
    """ Produce item_rec / user_rec metrics;
    then sweeps through multipliers for relevance-diversity curve,
    interpreting mult<1 as item min-exposure and mult>=1 as user max-limit
    """
    def __init__(self, D, V,
        mult=[], # [0, 0.1, 0.2, 0.5, 1, 3, 10, 30, 100],
        models_to_run=[
            "Rand", "Pop",
            "Hawkes", "HP",
            "RNN", "RNN-Pop",
            "RNN-Hawkes", "RNN-HP",
            "EMA", "RNN-EMA",
            "ALS", "LogisticMF",
            "BPR-Item", "BPR-User",
            ],
        model_hyps={},
        device="cuda" if torch.cuda.is_available() else "cpu",
        cvx=False,
        online=False,
        **mtch_kw
        ):
        self.D = D
        self.V = V

        self.mult = mult
        self.models_to_run = models_to_run
        self.model_hyps = model_hyps
        self.device = device

        if online:
            if not cvx:
                warnings.warn("online requires cvx, resetting cvx to True")
                cvx = True
            assert V is not None, "online cvx is trained with explicit valid_mat"

        self.mtch_kw = mtch_kw

        self.results = ExperimentResult(
            cvx, online,
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


    def metrics_update(self, name, S, T=None):
        target_csr = df_to_coo(self.D.target_df)
        score_mat = S.values

        if self.online:
            valid_mat = T.values
        elif self.cvx:
            valid_mat = score_mat
        else:
            valid_mat = None

        self.item_rec[name] = evaluate_item_rec(
            target_csr, score_mat, self._k1, device=self.device)
        self.user_rec[name] = evaluate_user_rec(
            target_csr, score_mat, self._c1, device=self.device)

        print(pd.DataFrame({
            'item_rec': self.item_rec[name],
            'user_rec': self.user_rec[name],
            }).T)

        if len(self.mult):
            self.mtch_[name] = self._mtch_update(target_csr, score_mat, valid_mat, name)


    def _mtch_update(self, target_csr, score_mat, valid_mat, name):
        """ assign user/item matches and return evaluation results.
        """
        confs = []
        for m in self.mult:
            if m < 1:
                # lower-bound is interpreted as item min-exposure
                confs.append((self._k1, self._c1 * m, 'lb'))
            else:
                # upper-bound is interpreted as user max-limit
                confs.append((self._k1 * m, self._c1, 'ub'))

        mtch_kw = self.mtch_kw.copy()
        if self.cvx:
            mtch_kw['valid_mat'] = valid_mat
            mtch_kw['prefix'] = f"{name}-{self.online}"
        else:
            mtch_kw['argsort_ij'] = _argsort(score_mat, device=self.device)

        out = []
        for k, c, constraint_type in confs:
            res = evaluate_mtch(
                target_csr, score_mat, k, c, constraint_type=constraint_type,
                cvx=self.cvx, device=self.device, **mtch_kw
            )
            res.update({'k': k, 'c': c})
            out.append(res)

        return out


    def transform(self, model, D):
        if model == "Rand":
            return Rand().transform(D)

        if model == "Pop":
            return self._pop.transform(D)

        if model == "EMA":
            return EMA(D.horizon).transform(D) * self._pop_item.transform(D)

        if model == "Hawkes":
            return self._hawkes.transform(D) * self._pop_item.transform(D)

        if model == "HP":
            return self._hawkes_poisson.transform(D) * self._pop_item.transform(D)

        if model == "RNN":
            return self._rnn.transform(D)

        if model == "RNN-Pop":
            return self._rnn.transform(D) * Pop(1, 0).transform(D)

        if model == "RNN-EMA":
            return self._rnn.transform(D) * EMA(D.horizon).transform(D)

        if model == "RNN-Hawkes":
            return self._rnn.transform(D) * self._hawkes.transform(D)

        if model == "RNN-HP":
            return self._rnn.transform(D) * self._hawkes_poisson.transform(D)

        if model == "BPR-Item":
            return self._bpr_item.transform(D)

        if model == "BPR-User":
            return self._bpr_user.transform(D)

        if model == "ALS":
            return self._als.transform(D)

        if model == "LogisticMF":
            return self._logistic_mf.transform(D)


    def run(self):
        for model in self.models_to_run:
            print("running", model)
            S = self.transform(model, self.D)

            if self.D.mask_df is not None:
                S = S + self.D.mask_df

            if self.online:
                V = self.V.reindex(self.D.item_in_test.index, axis=1)
                T = self.transform(model, V)

                if self.V.mask_df is not None:
                    T = T + self.V.mask_df
            else:
                T = None
            self.metrics_update(model, S, T)


    @cached_property
    def _pop(self):
        return Pop().fit(self.D.training_data)

    @cached_property
    def _pop_item(self):
        return Pop(user_rec=False, item_rec=True).fit(self.D.training_data)

    @cached_property
    def _rnn(self):
        fitted = RNN(self.D.training_data.item_df,
            **self.model_hyps.get("RNN", {})
        ).fit(self.D.training_data)
        for name, param in fitted.model.named_parameters():
            print(name, param.data.shape)
        return fitted

    @cached_property
    def _hawkes(self):
        return Hawkes(self.D.horizon).fit(self.D.training_data)

    @cached_property
    def _hawkes_poisson(self):
        return HawkesPoisson(self._hawkes).fit(self.V)

    @cached_property
    def _bpr_item(self):
        return LightFM_BPR(item_rec=True).fit(self.D.training_data)

    @cached_property
    def _bpr_user(self):
        return LightFM_BPR(user_rec=True).fit(self.D.training_data)

    @cached_property
    def _als(self):
        return ALS().fit(self.D.training_data)

    @cached_property
    def _logistic_mf(self):
        return LogisticMF().fit(self.D.training_data)


def main(name, *args, **kw):
    prepare_fn = getattr(dataset, name)
    D, V = prepare_fn(*args)
    self = Experiment(D, V, **kw)
    self.run()
    self.results.print_results()
    return self


def plot_results(self, logy=True):
    """ self is an instance of Experiment or ExperimentResult """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))
    df = [self.get_mtch_(k=self._k1), self.get_mtch_(c=self._c1)]

    xname = [f'ItemRec Prec@{self._k1}', f'UserRec Prec@{self._c1}']
    yname = ['item_ppl', 'user_ppl']

    for ax, df, xname, yname in zip(ax, df, xname, yname):
        ax.set_prop_cycle('color', [
            plt.get_cmap('tab20')(i/20) for i in range(20)])
        if df is not None:
            ax.plot(
                df.loc['prec'].unstack().values.T,
                df.loc[yname].unstack().values.T,
                '.-',
            )
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        if logy:
            ax.set_yscale('log')
    fig.legend(
        df.loc['prec'].unstack().index.values,
        bbox_to_anchor=(0.1, 0.9, 0.8, 0), loc=3, ncol=4,
        mode="expand", borderaxespad=0.)
    fig.subplots_adjust(wspace=0.25)
    return fig
