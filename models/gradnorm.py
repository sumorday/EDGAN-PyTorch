import torch


def penalty_normalize_gradient(net_D, x, **kwargs):
    """
                          1 - f
    f_hat = -------------------------------
               ||1 - grad_f ||+ |1 - f|
    """
    x.requires_grad_(True)
    f = net_D(x, **kwargs)
    grad = torch.autograd.grad(
        f, [x], torch.ones_like(f), create_graph=True, retain_graph=True)[0]
    grad_norm = torch.norm(torch.flatten(1 - grad, start_dim=1), p=2, dim=1)
    grad_norm = grad_norm.view(-1, *[1 for _ in range(len(f.shape) - 1)])
    f_hat = ((1 - f) / (grad_norm + torch.abs(1 - f)))
    return f_hat
