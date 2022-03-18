import pandas as pd, numpy as np
import functools
from ..util import timed, LazyDenseMatrix

from tick.hawkes import HawkesSumExpKern


class Hawkes:
    def __init__(self, horizon, scales=np.logspace(-6, 1), C=1e3,
                 training_eps=0, hetero=True, max_iter=1000):

        self.model = HawkesSumExpKern(1. / scales, C=C, verbose=True, max_iter=max_iter)
        self._input_fn = functools.partial(
            _input_fn, horizon=horizon, training_eps=training_eps, hetero=hetero)
        self.hetero = hetero

    @timed("Hawkes.fit", inline=False)
    def fit(self, D):
        training_user = D.user_df[
            (D.user_df['_hist_ts'].apply(len) > 0) &
            (D.user_df['TEST_START_TIME'] < np.inf)  # training users must have finite time
        ]
        X = training_user.apply(lambda x: self._input_fn(
            x['_hist_ts'], x['TEST_START_TIME'], training=True), axis=1).tolist()

        self.model.fit(X)
        self._learned_coeffs = _get_learned_coeffs(self.model)
        print(pd.DataFrame(self._learned_coeffs))
        return self

    def transform(self, D, state_only=False):
        X = D.user_in_test.apply(lambda x: self._input_fn(
            x['_hist_ts'], x['TEST_START_TIME'], training=False), axis=1).tolist()

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
            item_ones = np.ones(len(D.item_in_test))
            return (LazyDenseMatrix(np.log(user_intensities)[:, None]) @
                    LazyDenseMatrix(item_ones[:, None]).T).exp()


def _input_fn(hist_ts, test_start_time, horizon, training, training_eps, hetero):
    """ format to data and ctrl channels relative to the first observation """
    if len(hist_ts):
        data = (np.array(hist_ts[1:]) - hist_ts[0]) / horizon
        end_time = (test_start_time - hist_ts[0]) / horizon
    else:  # users without histories will receive baseline intensity predictions
        data = np.array([], dtype=np.asarray(test_start_time).dtype)
        end_time = float('inf')

    ctrl = np.array([0.0, end_time]) if hetero else np.array([end_time])

    if training:
        if training_eps > 0:
            data = np.sort(data + np.random.rand(len(data)) * training_eps)
            end_time = end_time + training_eps
        return [data, ctrl]
    else:
        return [data, ctrl[:-1]], end_time


def _get_learned_coeffs(model):
    return {
        'scales': 1. / model.decays,
        'x_by_x': model.adjacency[0][0],
        'x_by_s': model.adjacency[0][1],
        'x_base': model.baseline[0],
    }


def _predict_fn(x, end_time, decays):
    decays = decays.reshape((-1, 1))
    h_by_x = decays * np.exp(- decays * (end_time - x.reshape((1, -1))))
    h_by_s = decays * np.exp(- decays * end_time)
    return np.hstack([
        h_by_x.sum(axis=1),  # * model.adjacency[0][0],
        h_by_s.sum(axis=1),  # * model.adjacency[0][1] * hetero,
        1,  # model.baseline[0]
    ])


def _verify_estimated_intensity(model, X, user_intensities):
    print("verifying estimated intensity")
    for i, (x, y) in enumerate(zip(X, user_intensities)):
        p = model.estimated_intensity(x[0], x[1] * (1 - 1e-10), x[1])[0][0][-1]
        assert np.allclose(p, y)
    print(f"verified estimated intensity for {len(X)} cases")
