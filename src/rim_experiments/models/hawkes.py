import pandas as pd, numpy as np
import functools
from ..util import timed, LowRankDataFrame

from tick.hawkes import HawkesSumExpKern


class Hawkes:
    def __init__(self, horizon, scales=np.logspace(-6, 1), C=1e3,
        training_eps=0, hetero=True, max_iter=1000):

        self.model = HawkesSumExpKern(1./scales, C=C, verbose=True, max_iter=max_iter)
        self._input_fn = functools.partial(_input_fn,
            horizon=horizon, training_eps=training_eps, hetero=hetero)
        self.hetero = hetero

    @timed("Hawkes.fit")
    def fit(self, D):
        input_fn = functools.partial(self._input_fn, training=True)
        training_user = D.user_df[
            (D.user_df['_hist_len']>0) &
            (D.user_df['TEST_START_TIME'] < np.inf)
        ] # _timestamps includes TEST_START_TIME
        X = list(map(input_fn, training_user['_timestamps'].values))

        self.model.fit(X)
        self._learned_coeffs = _get_learned_coeffs(self.model)
        print(pd.DataFrame(self._learned_coeffs))
        return self

    @functools.lru_cache(1)
    def transform(self, D, state_only=False):
        input_fn = functools.partial(self._input_fn, training=False)
        X = list(map(input_fn, D.user_in_test['_timestamps'].values))

        predict_fn = functools.partial(_predict_fn, decays=self.model.decays)
        user_states = np.vstack([predict_fn(x[0], t) for (x, t) in X])

        if state_only:
            return pd.Series(user_states.tolist(), index=D.user_in_test.index)
        else:
            user_intensities = user_states @ np.hstack([
                self._learned_coeffs['x_by_x'],
                self._learned_coeffs['x_by_s'] * self.hetero,
                self._learned_coeffs['x_base'],
            ])
            if hasattr(D, '_is_synthetic_data') and D._is_synthetic_data:
                _verify_estimated_intensity(self.model, X, user_intensities)
            return LowRankDataFrame(
                np.log(user_intensities)[:, None], np.ones(len(D.item_df))[:, None],
                index=D.user_in_test.index, columns=D.item_df.index, act='exp')


def _input_fn(raw_ts, horizon, training, training_eps, hetero):
    """ format to data and ctrl channels relative to the first observation """
    data = (np.array(raw_ts[1:-1]) - raw_ts[0]) / horizon
    end_time = (raw_ts[-1] - raw_ts[0]) / horizon

    ctrl = np.array([0.0, end_time]) if hetero else np.array([end_time])

    if training:
        if training_eps>0:
            data = np.sort(data + np.random.rand(len(data)) * training_eps)
            end_time = end_time + training_eps
        return [data, ctrl]
    else:
        return [data, ctrl[:-1]], end_time


def _get_learned_coeffs(model):
    return {
        'scales': 1./model.decays,
        'x_by_x': model.adjacency[0][0],
        'x_by_s': model.adjacency[0][1],
        'x_base': model.baseline[0],
    }


def _predict_fn(x, end_time, decays):
    decays = decays.reshape((-1, 1))
    h_by_x = decays * np.exp(- decays * (end_time - x.reshape((1, -1))))
    h_by_s = decays * np.exp(- decays * end_time)
    return np.hstack([
        h_by_x.sum(axis=1), # * model.adjacency[0][0],
        h_by_s.sum(axis=1), # * model.adjacency[0][1] * hetero,
        1, # model.baseline[0]
    ])


def _verify_estimated_intensity(model, X, user_intensities):
    print("verifying estimated intensity")
    for i, (x, y) in enumerate(zip(X, user_intensities)):
        p = model.estimated_intensity(x[0], x[1]*(1-1e-10), x[1])[0][0][-1]
        assert np.allclose(p, y)
    print(f"verified estimated intensity for {len(X)} cases")
