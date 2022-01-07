import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_rec_results(self, metric_name='recall'):
    """ self is an instance of Experiment or ExperimentResult """
    ir = pd.DataFrame(self.item_rec).T
    ur = pd.DataFrame(self.user_rec).T
    df = ir[[metric_name]] * 100
    axname_itemrec_recall = f"Recommendation (ItemRec) {metric_name}@{self._k1} (x100)"
    axname_userrec_recall = f'Marketing (UserRec) {metric_name}@{self._c1} (x100)'

    df = df.rename(columns={metric_name: axname_itemrec_recall})
    df[axname_userrec_recall] = ur[metric_name] * 100
    df = df.reset_index()
    df['index'] = df['index'].apply(lambda x: x.replace('-Extra', '_ex').replace('-Item', '_item')
                                               .replace('-User', '_user').replace('-Base', '')
                                               .replace('-0', '_0').replace('-1', '_1'))
    df['index'] = df['index'].apply(lambda x: 'ItemPopularity-' + x if x in ['Hawkes', 'HP', 'EMA'] else x)
    df['index'] = df['index'].apply(lambda x: 'ItemPopularity-UserPopularity' if x == 'Pop' else x)
    df['index'] = df['index'].apply(lambda x: x.replace('Rand', 'Random-None'))
    df['index'] = df['index'].apply(lambda x: x.replace('-Pop', '-UserPopularity').replace('HP', 'HawkesPoisson'))
    df['Base model'] = df['index'].apply(lambda x: x.split('-')[0])
    df['Intensity modeling'] = df['index'].apply(lambda x: x.split('-')[1] if '-' in x else 'None')
    sns.set(rc={'figure.figsize': (8, 5), "font.size": 20, "axes.titlesize": 16,
                "axes.labelsize": 16, "xtick.labelsize": 10, "ytick.labelsize": 10},
            style="white")
    markers = {
        'ItemPopularity': 'X',
        'Transformer': 'o',
        'RNN': 'P',
        'Random': '$RND$',
        'BPR': '$BPR$',
        'GraphConv': '$GC$',
        'GraphConv_ex': '$GC\'$',
        'ALS': '$ALS$',
        'LogisticMF': '$LMF$',
        'LDA': '$LDA$',
        'BPR_item': '$BPR_i$',
        'BPR_user': '$BPR_u$'}
    large = 3000
    big = 1000
    sm = 80
    sizes = {
        'ItemPopularity': sm,
        'Transformer': sm,
        'RNN': sm,
        'Random': big,
        'BPR': big,
        'GraphConv': 600,
        'GraphConv_ex': big,
        'ALS': big,
        'LogisticMF': big,
        'LDA': big,
        'BPR_item': big,
        'BPR_user': big}
    markers.update({x: f"${x}$" for x in df['Base model'] if x not in markers})
    sizes.update({x: large for x in df['Base model'] if x not in sizes})
    figure = sns.relplot(
        x=axname_itemrec_recall, y=axname_userrec_recall, style='Base model', size='Base model', sizes=sizes,
        hue_order=['UserPopularity', 'EMA', 'HawkesPoisson', 'Hawkes', 'None'], markers=markers,
        linewidth=0.25, style_order=list(sizes.keys()), size_order=list(sizes.keys()),
        hue='Intensity modeling', data=df, facet_kws={'sharex': False, 'sharey': False}, legend='full')
    ax = figure.fig.axes[0]
    pop = df[df['index'] == 'ItemPopularity-UserPopularity'].iloc[0]
    ax.axvline(x=pop[axname_itemrec_recall], c='grey', linestyle='--', alpha=0.6, lw=1)
    ax.axhline(y=pop[axname_userrec_recall], c='grey', linestyle='--', alpha=0.6, lw=1, label='popularity baseline')
    return figure


def plot_mtch_results(self, logy=True):
    """ self is an instance of Experiment or ExperimentResult """
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))
    df = [self.get_mtch_(k=self._k1), self.get_mtch_(c=self._c1)]

    xname = [f'ItemRec Prec@{self._k1}', f'UserRec Prec@{self._c1}']
    yname = ['item_ppl', 'user_ppl']

    for ax, df, xname, yname in zip(ax, df, xname, yname):
        ax.set_prop_cycle('color', [
            plt.get_cmap('tab20')(i / 20) for i in range(20)])
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
        ax.axhline(getattr(self, yname + '_baseline'), ls='-.', color='gray')
    fig.legend(
        df.loc['prec'].unstack().index.values.tolist() + [yname + '_baseline'],
        bbox_to_anchor=(0.1, 0.9, 0.8, 0), loc=3, ncol=3,
        mode="expand", borderaxespad=0.)
    fig.subplots_adjust(wspace=0.25)
    return fig
