import torch
import stochman 

def geodesic_minimizing_energy_with_library(curve, 
                                            manifold, 
                                            library_size,
                                            optimizer=torch.optim.Adam, 
                                            max_iter=150, 
                                            eval_grid=20, 
                                            lr=1e-2, 
                                            minimize_energy=True, 
                                            return_losses=True):
    """
    Compute a geodesic curve connecting two points by minimizing its energy.

    Mandatory inputs:
        curve:      A curve object representing a curve with fixed end-points.
                    When the function returns, this object has been updated to
                    be a geodesic curve.
        manifold:   A manifold object representing the space over which the
                    geodesic is defined. This object must provide a
                    'curve_energy' function through which pytorch can
                    back-propagate.

    Optional inputs:
        optimizer:  Choice of iterative optimizer.
                    Default: torch.optim.Adam
        max_iter:   The maximum number of iterations of the optimizer.
                    Default: 150
        eval_grid:  The number of points along the curve where
                    energy is evaluated.
                    Default: 20

    Output:
        success:    True if the algorithm converged, False otherwise.

    Example usage:
    S = Sphere()
    p0 = torch.tensor([0.1, 0.1]).reshape((1, -1))
    p1 = torch.tensor([0.3, 0.7]).reshape((1, -1))
    C = CubicSpline(begin=p0, end=p1, num_nodes=8, requires_grad=True)
    geodesic_minimizing_energy(C, S)
    """
    losses = []
    # Initialize optimizer and set up closure
    if type(library_size) != torch.tensor:
        library_size = torch.tensor(library_size)
        
    alpha = torch.linspace(0, 1, eval_grid, dtype=curve.begin.dtype, device=curve.device)
    opt = optimizer(curve.parameters(), lr=lr)

    def closure():
        opt.zero_grad()
        if minimize_energy: 
            loss = manifold.curve_energy(curve(alpha), library_size).mean()
        else:
            loss = manifold.curve_length(curve(alpha), library_size).mean()
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
