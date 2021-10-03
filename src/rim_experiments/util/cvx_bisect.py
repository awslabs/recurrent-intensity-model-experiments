""" 
max_pi  E_xy[s(x,y) * pi(x,y)]
s.t.    E_y[pi(x,y)] <= alpha(x)
        E_x[pi(x,y)] >= beta(y)
"""

import numpy as np
import torch, warnings


def lagrangian(pi, u, v, s, alpha, beta, eps):
    """
    L(pi, u, v; s, alpha, beta, eps)
        = E_xy[ s(x,y) * pi(x,y) - u(x)(pi(x,y)-alpha(x)) - v(y)(pi(x,y)-beta(y)) ]
        + eps * E_xy[H(pi)]
    """
    grad_u = alpha - pi.mean(axis=1)
    grad_v = beta - pi.mean(axis=0)
    ent = - pi * pi.clip(1e-10, None).log() - (1-pi) * (1-pi).clip(1e-10, None).log()
    return (s * pi).mean() + (u * grad_u).mean() + (v * grad_v).mean() + eps * ent.mean()


def primal_solution(u, v, s, eps):
    """
    max_pi L(pi, u, v; ...) solved by
    pi = sigmoid[(s(x,y) - u(x) - v(y)) / eps]
    """
    u = torch.as_tensor(u).to(s.device)
    v = torch.as_tensor(v).to(s.device)
    r = s - u[:, None] - v[None, :]
    if eps > 0:
        return torch.sigmoid(r / eps)
    else:
        # TODO: how to represent subgradients in [0, 1]?
        return torch.sign(r) * 0.5 + 0.5


def dual_complete(u, v, s, alpha, beta, eps):
    """
    min_{u>=0, v<=0} d(u, v)
        = E_xy [ u(x)alpha(x) + v(y)beta(y) + Softplus(1/eps)(s-u-v) ]
    """
    u = torch.as_tensor(u).to(s.device)
    v = torch.as_tensor(v).to(s.device)
    if eps > 0:
        sp = torch.nn.Softplus(1./eps)(s - u[:, None] - v[None, :])
    else:
        sp = torch.nn.ReLU()(s - u[:, None] - v[None, :])
    return (u * alpha).mean() + (v * beta).mean() + sp.mean()


def grad_u(u, v, s, alpha, eps, stable=True):
    """
    find u = min{u>=0 : E_y[pi(x,y)] <= alpha(x)}
    """
    if stable: # this does not make a practical difference wrt. slow convergence
        u = torch.as_tensor(u).to(s.device)[:, None]
        v = torch.as_tensor(v).to(s.device)[None, :]
        alpha = torch.as_tensor(alpha)

        z = alpha.log() - (1-alpha).log()
        r = s - v
        d = (u - r) / eps + z    # u = r - z * eps, if |Y|=1

        d_lt_0 = ( d.exp() - 1 ) / (
            (z.exp() + 1) * (((u - r) / eps).exp() + 1)
        ) # grad < 0

        d_gt_0 = ( 1 - (-d).exp() ) / (
            ((-z).exp() + 1) * (((r - u) / eps).exp() + 1)
        ) # grad > 0

        return torch.where(d < 0, d_lt_0, d_gt_0).mean(axis=1)
    else:
        pi = primal_solution(u, v, s, eps)
        return alpha - pi.mean(axis=1)


def dual_solve_u(v, s, alpha, eps, verbose=False):
    """
    min_{u>=0} max_pi L(pi, u, v)
        = E_xy [ u(x)alpha(x) + v(y)beta(y) + Softplus(1/eps)(s-u-v) ],
            where u = min{u>=0 : E_y[pi(x,y)] <= alpha(x)}
    find exact u s.t. E_y[pi(x,y)] == alpha(x)
    """
    if alpha < 0 or alpha > 1:
        warnings.warn(f"clipping alpha={alpha} to [0, 1]")
        alpha = np.clip(alpha, 0, 1)

    v = torch.as_tensor(v).to(s.device)
    alpha = torch.as_tensor(alpha).to(s.device)
    eps = torch.as_tensor(eps).to(s.device)

    z = alpha.log() - (1-alpha).log()

    if alpha == 0 or alpha == 1: # z = +-infinity
        u = -z * torch.ones_like(s[:, 0])
        return u

    u_min = (s - v[None, :]).amin(axis=1) - z * eps - 1e-3
    u_max = (s - v[None, :]).amax(axis=1) - z * eps + 1e-3

    assert (grad_u(u_min, v, s, alpha, eps) <= 0).all()
    assert (grad_u(u_max, v, s, alpha, eps) >= 0).all()

    for i in range(50):
        u = (u_min + u_max) / 2
        g = grad_u(u, v, s, alpha, eps)
        assert not u.isnan().any()
        u_min = torch.where(g<0, u, u_min)
        u_max = torch.where(g>0, u, u_max)

    return u


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
    u = dual_solve_u(v, s, alpha, eps)
    u = dual_clip(u, constraint_type)
    return dual_complete(u, v, s, alpha, beta, eps)


def dual_iterate(v, s, alpha, beta, eps,
    constraint_type_a='ub', constraint_type_b='eq',
    max_iters=10, stepsize=0):
    for epoch in range(max_iters):
        u = dual_solve_u(v, s, alpha, eps)
        u = dual_clip(u, constraint_type_a)

        yield v, dual_complete(u, v, s, alpha, beta, eps)

        if stepsize > 0:
            grad_v = grad_u(v, u, s.T, beta, eps)
            v = v - grad_v * stepsize
        else:
            v = dual_solve_u(u, s.T, beta, eps)
        v = dual_clip(v, constraint_type_b)


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
            u = dual_solve_u(v, s, alpha, eps)
            u = dual_clip(u, 'ub')
            y = dual_complete(u, v, s, alpha, beta, eps)
            arr.append([v.numpy(), y.numpy()])
            print(eps, u, v, y)

            v = dual_solve_u(u, s.T, beta, eps)

            u = dual_solve_u(v, s, alpha, eps)
            u = dual_clip(u, 'ub')
            y = dual_complete(u, v, s, alpha, beta, eps)
            arr.append([v.numpy(), y.numpy()])
            print(eps, u, v, y)

    pl.plot(*zip(*arr), '+--', label='annealed')

    pl.ylabel("dual objective", fontsize=12)
    pl.xlabel("dual variable v", fontsize=12)
    pl.legend(loc='upper right')

    fig.savefig('cvx_synthetic.pdf', bbox_inches='tight')
