""" Detailed steps to solve the following convex optimization problem.

    max_pi  E_xy[s(x,y) * pi(x,y)]
    s.t.    E_y[pi(x,y)] <= alpha(x)
            E_x[pi(x,y)] >= beta(y)
"""

import numpy as np
import torch, os, torch.nn.functional as F


def lagrangian(pi, u, v, s, alpha, beta, eps):
    """
    L(pi, u, v; s, alpha, beta, eps)
        = E_xy[ s(x,y) * pi(x,y) - u(x)(pi(x,y)-alpha(x)) - v(y)(pi(x,y)-beta(y)) ]
        + eps * E_xy[H(pi)]
    """
    grad_u = alpha - pi.mean(1)
    grad_v = beta - pi.mean(0)
    ent = - pi * pi.clip(1e-10, None).log() - (1 - pi) * (1 - pi).clip(1e-10, None).log()
    return (s * pi).mean() + (u * grad_u).mean() + (v * grad_v).mean() + eps * ent.mean()


def primal_solution(s, u=None, v=None, eps=1):
    """
    max_pi L(pi, u, v; ...) solved by
    pi = sigmoid[(s(x,y) - u(x) - v(y)) / eps]
    """
    if u is not None:
        s = s - torch.as_tensor(u, device=s.device).reshape((-1, 1))
    if v is not None:
        s = s - torch.as_tensor(v, device=s.device).reshape((1, -1))

    if eps > 0:
        return torch.sigmoid(s if eps == 1 else s / eps)
    else:
        # obtain subgradients via torch.floor or ceil
        return torch.sign(s) * 0.5 + 0.5


def dual_complete(u, v, s, alpha, beta, eps):
    """
    min_{u>=0, v<=0} d(u, v)
        = E_xy [ u(x)alpha(x) + v(y)beta(y) + Softplus(1/eps)(s-u-v) ]
    """
    u = torch.as_tensor(u, device=s.device).reshape((-1, 1))
    v = torch.as_tensor(v, device=s.device).reshape((1, -1))
    if eps > 0:
        sp = torch.nn.Softplus(1. / eps)(s - u - v)
    else:
        sp = torch.nn.ReLU()(s - u - v)
    return (u * alpha).mean() + (v * beta).mean() + sp.mean()


def grad_u(s, alpha, eps):
    """
    alpha - E_y[sigmoid(s_xy / eps)], where

    sigmoid(c) - sigmoid(z) = 1 / (1+e^-c) - 1 / (1+e^-z)
                            = (e^-z - e^-c) / (1+e^-c) / (1+e^-z)
                            = (1 - e^(z-c)) / (1+e^-c) / (1+e^z)
    assume 0 <= alpha <= 1 and |z| < inf
    """
    if eps == 0:
        pi = primal_solution(s, eps=eps)
        return _subgradient(alpha, pi)
    elif int(os.environ.get('CVX_STABLE', 0)):
        alpha = torch.as_tensor(alpha, device=s.device).clip(0, 1)
        c = alpha.log() - (1 - alpha).log()
        c = c.reshape((-1, 1))
        sgn = torch.sign(c * eps - s)
        alpha_pi = sgn * (
            torch.log1p(-torch.exp(-torch.abs(c - s / eps)))
            - F.softplus(-sgn * c) - F.softplus(sgn * s / eps)
        ).exp()
        return alpha_pi.mean(1)
    else:
        pi = primal_solution(s, eps=eps)
        return alpha - pi.mean(1)


def _subgradient(alpha, pi):
    lb = alpha - torch.ceil(pi).mean(1)
    ub = alpha - torch.floor(pi).mean(1)
    return torch.where(ub < 0, ub, torch.where(
        lb > 0, lb, torch.zeros_like(lb)))


@torch.no_grad()
def dual_solve_u(s, alpha, eps, verbose=False, n_iters=100, gtol=0, s_guess=None):
    """
    find exact u s.t. E_y[pi(x,y)] == alpha(x); transpose s to solve v

    Note: provide s_guess when exclude_train=True to trim the search space
    """
    alpha = torch.as_tensor(alpha, device=s.device)
    if alpha.max() <= 0 or alpha.min() >= 1:
        c = torch.sign(alpha - 0.5) * np.inf
        u = -c * torch.ones_like(s[:, 0])
        return u, 0

    c = alpha.log() - (1 - alpha).log()  # |c| < inf
    u_min = s.amin(1) - c * eps - 1e-2
    u_max = s.amax(1) - c * eps + 1e-2
    u_guess = []
    if s_guess is not None:
        u_guess.append(torch.as_tensor(s_guess).to(s) - c * eps - 1e-2)
    # k = (alpha * s.shape[1] + 1).clip(None, s.shape[1]).int()
    # u_guess.append(s.topk(k).values[:, -3:].T)

    assert (grad_u(s - u_min.reshape((-1, 1)), alpha, eps) <= 0).all()
    assert (grad_u(s - u_max.reshape((-1, 1)), alpha, eps) >= 0).all()

    for i in range(n_iters):
        if i < len(u_guess):
            u = u_guess[i]
        else:
            u = (u_min + u_max) / 2
        g = grad_u(s - u.reshape((-1, 1)), alpha, eps)
        assert not u.isnan().any()
        if g.abs().max() < gtol:
            break
        u_min = torch.where(g < 0, u, u_min)
        u_max = torch.where(g > 0, u, u_max)

    return u, (i + 1)


def dual_clip(u, constraint_type):
    if constraint_type == 'ub':
        u = u.clip(0, None)
    elif constraint_type == 'lb':
        u = u.clip(None, 0)
    # else: eq
    return u


### the following is mostly used for visualization

def dual_v(v, s, alpha, beta, eps, constraint_type='ub'):
    """
    min_{v<=0} d(v)
        = min_{u>=0} max_pi L(pi, u, v)
        = E_xy [ u(x)alpha(x) + v(y)beta(y) + Softplus(1/eps)(s-u-v) ],
            where u = min{u>=0 : E_y[pi(x,y)] <= alpha(x)}

    When eps -> 0, Softplus(1/eps)(z) -> max(z, 0) -> z * sigmoid(z / eps) = z * pi

    d(v) = E_xy [ u(x)alpha(x) + v(y)beta(y) + (s - u - v) pi ]
         = E_xy [ s(x,y) pi(x,y) + u(x)(alpha(x) - pi(x,y)) + v(y)(beta(y) - pi(x,y)) ]
         >= E_xy [ s(x,y) + v(y)(beta(y) - pi(x,y)) ]
    """
    u, _ = dual_solve_u(s - v.reshape((1, -1)), alpha, eps)
    u = dual_clip(u, constraint_type)
    return dual_complete(u, v, s, alpha, beta, eps)


def dual_iterate(v, s, alpha, beta, eps,
                 constraint_type_a='ub', constraint_type_b='eq',
                 max_iters=10, stepsize=0):
    for epoch in range(max_iters):
        u, _ = dual_solve_u(s - v.reshape((1, -1)), alpha, eps)
        u = dual_clip(u, constraint_type_a)

        yield v, dual_complete(u, v, s, alpha, beta, eps), primal_solution(s, u, v, eps)

        if stepsize > 0:
            grad_v = grad_u((s - u.reshape((-1, 1)) - v.reshape((1, -1))).T, beta, eps)
            v = v - grad_v * stepsize
        else:
            v, _ = dual_solve_u((s - u.reshape((-1, 1))).T, beta, eps)
        v = dual_clip(v, constraint_type_b)


if __name__ == '__main__':
    import matplotlib, pylab as pl
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    s = torch.tensor([[1], [0.5]])
    alpha = 1
    beta = 0.4
    eps = 1

    fig, ax = pl.subplots(figsize=(4, 3))
    v_list = np.linspace(-1, 3.8, 100)
    colors = []
    for i, eps in enumerate([1, 0.5, 0.01]):
        f = [dual_v(torch.as_tensor([v]), s, alpha, beta, eps).tolist()
             for v in v_list]
        p = pl.plot(v_list, f, ls=':', label=f'$\epsilon$={eps}')  # noqa: W605
        colors.append(p[0].get_color())

    for i, eps in enumerate([1, 0.5, 0.01]):
        pl.plot(
            *zip(*[
                (v.cpu().numpy(), y.cpu().numpy())
                for v, y, _ in dual_iterate(
                    torch.as_tensor([-0.5]), s, alpha, beta, eps, stepsize=2,
                )]),
            'o-', mfc='none', color=colors[i], label='(sub)gradient' if i == 2 else None
        )
        for v, y, _ in dual_iterate(
            torch.as_tensor([-0.5]), s, alpha, beta, eps, stepsize=2,
        ):
            print(eps, v, y)

    v = torch.as_tensor([-0.5])
    arr = []
    for eps in [1, 0.5, 0.01]:
        for _ in range(3):
            u, _ = dual_solve_u(s - v.reshape((1, -1)), alpha, eps)
            u = dual_clip(u, 'ub')
            y = dual_complete(u, v, s, alpha, beta, eps)
            arr.append([v.numpy(), y.numpy()])
            print(eps, u, v, y)

            v, _ = dual_solve_u(s.T - u.reshape((1, -1)), beta, eps)

            u, _ = dual_solve_u(s - v.reshape((1, -1)), alpha, eps)
            u = dual_clip(u, 'ub')
            y = dual_complete(u, v, s, alpha, beta, eps)
            arr.append([v.numpy(), y.numpy()])
            print(eps, u, v, y)

    pl.plot(*zip(*arr), '+--', label='annealed')

    pl.ylabel("dual objective", fontsize=12)
    pl.xlabel("dual variable v", fontsize=12)
    pl.legend(loc='upper right')

    fig.savefig('cvx_synthetic.pdf', bbox_inches='tight')
