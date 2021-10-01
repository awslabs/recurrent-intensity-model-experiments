#!/usr/bin/env python
# coding: utf-8

import pylab as pl
pl.plot()
pl.show()


from rim_experiments import *
from rim_experiments.dataset import *

D, V = prepare_ml_1m_data()

kw = {
    "lb_mult": [0.5, 0.2, 0.1, 0],
#     "models_to_run": ["Pop", "RNN-Pop"],
}

offline = Experiment(D, V, **kw).run()

cvx = Experiment(D, V, **kw, cvx=True)
cvx._pretrain_rnn = offline._rnn
cvx.run()

online = Experiment(D, V, **kw, cvx=True, online=True)
online._pretrain_rnn = offline._rnn
online.run()

offline_lb = offline.get_mtch_(k=offline._k1, name="lb_")
cvx_lb = cvx.get_mtch_(k=cvx._k1, name="lb_")
online_lb = online.get_mtch_(k=online._k1, name="lb_")


plot_names = {
    'Rand': ('Rand', '.'),
    'Pop':  ('Pop',  '*'),
    'Hawkes':  ('Hawkes',  '$h$'),
    'HP':  ('Hawkes-Poisson',  '$p$'),
    'BPR': ('BPR', 'x'),
    'RNN': ('RNN', '$1$'),
    'RNN-Pop': ('RNN-Pop', '$2$'),
    'RNN-Hawkes': ('RNN-Hawkes', '$3$'),
    'RNN-HP': ('RNN-HP', '$4$'),
}


fig, ax = pl.subplots(1, 3, figsize=(7, 2.5))
xname = f'ItemRec Prec@{online._k1}'
yname = 'item_ppl'

hdl = []
for i, (ax, lb) in enumerate(zip(ax, [offline_lb, cvx_lb, online_lb])):
    for name, (label, marker) in plot_names.items():
        if name == 'BPR':
            name = 'BPR-Item'
        if name not in lb:
            continue
        hdl.extend(
            ax.plot(lb.loc['prec'][name], lb.loc[yname][name],
                    label=label, marker=marker, ls=':')
        )
    ax.set_xlabel(xname)
    if i==0:
        ax.set_ylabel(yname)
fig.legend(
    hdl, [k for k,v in plot_names.values() if k in online_lb],
    bbox_to_anchor=(0.1, 0.9, 0.8, 0), loc=3, ncol=4,
    mode="expand", borderaxespad=0.)
fig.subplots_adjust(wspace=0.3)
fig.show()
