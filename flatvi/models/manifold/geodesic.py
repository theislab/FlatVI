import torch

def sc_geodesic_minimizing_energy(curve, 
                                    manifold,
                                    optimizer=torch.optim.Adam, 
                                    max_iter=150, 
                                    eval_grid=20, 
                                    lr=1e-2, 
                                    minimize_energy=True, 
                                    return_losses=True):
    """Fit the minimal length spline curve.

    Args:
        curve (torch.tensor): tensor containing the curve values
        manifold (Manifold): single-cell statistical manifold class 
        optimizer (torch.optim.optimizer, optional): optimizer. Defaults to torch.optim.Adam.
        max_iter (int, optional): maximum iteration. Defaults to 150.
        eval_grid (int, optional): number of discretization steps. Defaults to 20.
        lr (float, optional): learning rate. Defaults to 1e-2.
        minimize_energy (bool, optional): whether to minimize energy (false minimizes length). Defaults to True.
        return_losses (bool, optional): return losses. Defaults to True.

    Returns:
        [float, None]: losses
    """
    losses = []
    # Get the range of interpolation between observations 
    alpha = torch.linspace(0, 1, eval_grid, dtype=curve.begin.dtype, device=curve.device)
    opt = optimizer(curve.parameters(), lr=lr)

    def closure():
        opt.zero_grad()
        if minimize_energy: 
            loss = manifold.curve_energy(curve(alpha)).mean()
        else:
            loss = manifold.curve_length(curve(alpha)).mean()
        loss.backward()
        losses.append(loss.item())
        return loss

    thresh = 1e-4

    for _ in range(max_iter):
        opt.step(closure=closure)
        max_grad = max([p.grad.abs().max() for p in curve.parameters()])
        if max_grad < thresh:
            break
    if return_losses:
        return losses
    else:
        return None
