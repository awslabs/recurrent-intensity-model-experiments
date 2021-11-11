""" Detailed steps to solve the following convex optimization problem.

    max_pi  E_xy[s(x,y) * pi(x,y)]
    s.t.    E_y[pi(x,y)] <= alpha(x)
            E_x[pi(x,y)] >= beta(y)
"""

import numpy as np
import torch, warnings, os


def lagrangian(pi, u, v, s, alpha, beta, eps):
    """
    L(pi, u, v; s, alpha, beta, eps)
        = E_xy[ s(x,y) * pi(x,y) - u(x)(pi(x,y)-alpha(x)) - v(y)(pi(x,y)-beta(y)) ]
        + eps * E_xy[H(pi)]
    """
    grad_u = alpha - pi.mean(1)
    grad_v = beta - pi.mean(0)
    ent = - pi * pi.clip(1e-10, None).log() - (1-pi) * (1-pi).clip(1e-10, None).log()
    return (s * pi).mean() + (u * grad_u).mean() + (v * grad_v).mean() + eps * ent.mean()


def s_u_v(s, u, v):
    if u is not None:
        s = s - torch.as_tensor(u, device=s.device).reshape((-1, 1))
    if v is not None:
        s = s - torch.as_tensor(v, device=s.device).reshape((1, -1))
    return s


def primal_solution(u, v, s, eps):
    """
    max_pi L(pi, u, v; ...) solved by
    pi = sigmoid[(s(x,y) - u(x) - v(y)) / eps]
    """
    if eps > 0:
        return torch.sigmoid(s_u_v(s, u, v) / eps)
    else:
        # obtain subgradients via torch.floor or ceil
        return torch.sign(s_u_v(s, u, v)) * 0.5 + 0.5


def dual_complete(u, v, s, alpha, beta, eps):
    """
    min_{u>=0, v<=0} d(u, v)
        = E_xy [ u(x)alpha(x) + v(y)beta(y) + Softplus(1/eps)(s-u-v) ]
    """
    u = torch.as_tensor(u, device=s.device).reshape((-1, 1))
    v = torch.as_tensor(v, device=s.device).reshape((1, -1))
    if eps > 0:
        sp = torch.nn.Softplus(1./eps)(s - u - v)
    else:
        sp = torch.nn.ReLU()(s - u - v)
    return (u * alpha).mean() + (v * beta).mean() + sp.mean()


def grad_u(u, v, s, alpha, eps):
    """
    find u = min{u>=0 : E_y[pi(x,y)] <= alpha(x)}
    """
    pi = primal_solution(u, v, s, eps)
    if eps>0:
        return alpha - pi.mean(1)
    else:
        return _subgradient(alpha, pi)


def _subgradient(alpha, pi):
    lb = alpha - torch.ceil(pi).mean(1)
    ub = alpha - torch.floor(pi).mean(1)
    return torch.where(ub < 0, ub, torch.where(
        lb > 0, lb, torch.zeros_like(lb)))


@torch.no_grad()
def dual_solve_u(v, s, alpha, eps, verbose=False, n_iters=100, gtol=0):
    """
    min_{u>=0} max_pi L(pi, u, v)
        = E_xy [ u(x)alpha(x) + v(y)beta(y) + Softplus(1/eps)(s-u-v) ],
            where u = min{u>=0 : E_y[pi(x,y)] <= alpha(x)}
    find exact u s.t. E_y[pi(x,y)] == alpha(x)
    """
    if alpha < 0 or alpha > 1:
        warnings.warn(f"clipping alpha={alpha} to [0, 1]")
        alpha = np.clip(alpha, 0, 1)

    alpha = torch.as_tensor(alpha, device=s.device)
    eps = torch.as_tensor(eps, device=s.device)

    z = alpha.log() - (1-alpha).log()

    if alpha == 0 or alpha == 1: # z = +-infinity
        u = -z * torch.ones_like(s[:, 0])
        return u, 0

    if 'CVX_STABLE' in os.environ and int(os.environ['CVX_STABLE']):
        v = torch.as_tensor(v, device=s.device).reshape((1, -1))
    else:
        s = s_u_v(s, None, v)
        v = None

    u_min = s_u_v(s, None, v).amin(1) - z * eps - 1e-3
    u_max = s_u_v(s, None, v).amax(1) - z * eps + 1e-3

    assert (grad_u(u_min, v, s, alpha, eps) <= 0).all()
    assert (grad_u(u_max, v, s, alpha, eps) >= 0).all()

    for i in range(n_iters):
        u = (u_min + u_max) / 2
        g = grad_u(u, v, s, alpha, eps)
        assert not u.isnan().any()
        if g.abs().max() < gtol:
            break
        u_min = torch.where(g<0, u, u_min)
        u_max = torch.where(g>0, u, u_max)

    return u, i


def dual_clip(u, constraint_type):
    if constraint_type == 'ub':
        u = u.clip(0, None)
    elif constraint_type == 'lb':
        u = u.clip(None, 0)
    # else: eq
    return u


def dual(v, s, alpha, beta, eps, constraint_type='ub'):
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
    u, _ = dual_solve_u(v, s, alpha, eps)
    u = dual_clip(u, constraint_type)
    return dual_complete(u, v, s, alpha, beta, eps)


def dual_iterate(v, s, alpha, beta, eps,
    constraint_type_a='ub', constraint_type_b='eq',
    max_iters=10, stepsize=0):
    for epoch in range(max_iters):
        u, _ = dual_solve_u(v, s, alpha, eps)
        u = dual_clip(u, constraint_type_a)

        yield v, dual_complete(u, v, s, alpha, beta, eps), primal_solution(u, v, s, eps)

        if stepsize > 0:
            grad_v = grad_u(v, u, s.T, beta, eps)
            v = v - grad_v * stepsize
        else:
            v, _ = dual_solve_u(u, s.T, beta, eps)
        v = dual_clip(v, constraint_type_b)


if 'CVX_STABLE' in os.environ and int(os.environ['CVX_STABLE']):
    print("CVX_STABLE")

    def _log_diff_exp(a, b):
        """ e^a - e^b = (1 - e^(b-a)) e^a, if a>b
        """
        min = lambda: torch.fmin(a, b)
        max = lambda: torch.fmax(a, b)
        log = torch.log1p(-torch.exp(min() - max())) + max()
        sign = torch.sign(a - b)
        return sign, log


    def _log_diff_sigmoid(z, x):
        """ sigmoid(z) - sigmoid(x)
        = (e^-x - e^-z) / (1+e^-z) / (1+e^-x)
        = e^-z[-] / (e^-z[-] + e^-z[+]) - e^-x[-] / (e^-x[-] + e^-x[+])
        = (e^(-z[-]-x[+]) - e^(-x[-]-z[+])) / (e^-z[-] + e^-z[+]) / (e^-x[-] + e^-x[+])

        >>> z = torch.tensor([[-1e20], [0.], [1e20]])
        >>> x = torch.tensor([-np.inf, -1e20, 0, 1e20, np.inf])
        >>> assert (torch.sigmoid(z) - torch.sigmoid(x) ==
        ... _log_diff_sigmoid(z, x)[0] * _log_diff_sigmoid(z, x)[1].exp()).all()
        """
        z_pos = lambda: z.clip(0, None)
        z_neg = lambda: -z.clip(None, 0)
        x_pos = lambda: x.clip(0, None)
        x_neg = lambda: -x.clip(None, 0)

        sign, log_nom = _log_diff_exp( -z_neg() - x_pos(), -x_neg() - z_pos() )
        return sign, log_nom - \
            torch.logaddexp(-z_neg(), -z_pos()) - torch.logaddexp(-x_neg(), -x_pos())


    def grad_u(u, v, s, alpha, eps):
        """ alpha - pi.mean(1)
        be more precise if eps->0 while there are a lot of ties
        """
        if eps==0:
            pi = torch.sign(s_u_v(s, u, v)) * 0.5 + 0.5
            return _subgradient(alpha, pi)

        alpha = torch.as_tensor(alpha, device=s.device).clip(0, 1)

        z = alpha.log() - (1-alpha).log()
        x = s_u_v(s, u, v) / eps
        sign, log = _log_diff_sigmoid(z, x)

        pos = torch.logsumexp((sign>0).log() + log, dim=1)
        neg = torch.logsumexp((sign<0).log() + log, dim=1)

        sign, log = _log_diff_exp(pos, neg)
        return sign * log.exp() / s.shape[1]


if __name__ == '__main__':
    import matplotlib, pylab as pl
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    s = torch.tensor([[1], [0.5]])
    alpha = 1
    beta = 0.4
    eps = 1

    fig, ax = pl.subplots(figsize=(4,3))
    v_list = np.linspace(-1, 3.8, 100)
    colors = []
    for i, eps in enumerate([1, 0.5, 0.01]):
        f = [dual(torch.as_tensor([v]), s, alpha, beta, eps).tolist()
             for v in v_list]
        p = pl.plot(v_list, f, ls=':', label=f'$\epsilon$={eps}')
        colors.append(p[0].get_color())

    for i, eps in enumerate([1, 0.5, 0.01]):
        pl.plot(*zip(*[(v.cpu().numpy(), y.cpu().numpy())
          for v, y in dual_iterate(
              torch.as_tensor([-0.5]), s, alpha, beta, eps, stepsize=2,
          )]), 'o-', mfc='none', color=colors[i], label=f'(sub)gradient' if i==2 else None)
        for v, y in dual_iterate(
              torch.as_tensor([-0.5]), s, alpha, beta, eps, stepsize=2,
          ):
            print(eps, v, y)

    v = torch.as_tensor([-0.5])
    arr = []
    for eps in [1, 0.5, 0.01]:
        for _ in range(3):
            u, _ = dual_solve_u(v, s, alpha, eps)
            u = dual_clip(u, 'ub')
            y = dual_complete(u, v, s, alpha, beta, eps)
            arr.append([v.numpy(), y.numpy()])
            print(eps, u, v, y)

            v, _ = dual_solve_u(u, s.T, beta, eps)

            u, _ = dual_solve_u(v, s, alpha, eps)
            u = dual_clip(u, 'ub')
            y = dual_complete(u, v, s, alpha, beta, eps)
            arr.append([v.numpy(), y.numpy()])
            print(eps, u, v, y)

    pl.plot(*zip(*arr), '+--', label='annealed')

    pl.ylabel("dual objective", fontsize=12)
    pl.xlabel("dual variable v", fontsize=12)
    pl.legend(loc='upper right')

    fig.savefig('cvx_synthetic.pdf', bbox_inches='tight')
