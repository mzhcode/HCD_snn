import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def heaviside(x: torch.Tensor):
    return (x >= 0.0).to(x)


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class SigmoidSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, input, alpha, threshold
    ):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return heaviside(input - threshold)

    @staticmethod
    def backward(ctx, grad_output):
        grad_ = None
        if ctx.needs_input_grad[0]:
            sgax = (ctx.saved_tensors[0] * ctx.alpha).sigmoid_()
            grad_x = grad_output * (1.0 - sgax) * sgax * ctx.alpha
        return grad_x, None, None


class AtanSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = ctx.alpha / 2 / (
                    1 + math.pi / 2 * ctx.alpha * ctx.saved_tensors[0].pow_(2)
            ) * grad_output
        return grad_x, None


class RectangularSurrogate(torch.autograd.Function):
    r"""
    default alpha=0.8
    """

    @staticmethod
    def forward(ctx, input, threshold=0.5, alpha=0.8):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        ctx.alpha = alpha  # surrogate gradient function hyper-parameter
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = (2 * abs(input - ctx.threshold) < ctx.alpha) * 1. / ctx.alpha
        return grad_input * temp, None, None


class TriangularSurrogate(torch.autograd.Function):
    r"""
    default alpha=1.0
    """

    @staticmethod
    def forward(ctx, input, alpha=1.0):
        ctx.save_for_backward(input)
        ctx.alpha = alpha  # surrogate gradient function hyper-parameter
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = (1 / ctx.alpha) * (1 / ctx.alpha) * (
            (ctx.alpha - input.abs()).clamp(min=0)
        )
        return grad_input * temp, None


class ERFSurrogate(torch.autograd.Function):
    r"""
    """

    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input)
        ctx.alpha = alpha  # surrogate gradient function hyper-parameter
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = (- (input * ctx.alpha).pow_(2)).exp_() * (
                ctx.alpha / math.sqrt(math.pi)
        )
        return grad_output * temp, None


surrogate_fn = TriangularSurrogate.apply
class pa_lif_n(nn.Module):
    def __init__(self,
                 T: int = 4,
                 tau: float = 0.25,
                 threshold: float = 0.5,
                 dim: int = 25000):
        super().__init__()
        assert T is not None, "T must be not None!!!"
        self.T = T
        self.tau = tau
        self.threshold = nn.Parameter(torch.as_tensor(threshold)) if threshold is None else threshold
        self.surrogate_function = surrogate_fn
        self.surrogate_f = surrogate_fn
        self.mse = torch.nn.MSELoss(reduction="none")
        self.mem_loss = 0
        self.indices = torch.arange(self.T, device='cuda').unsqueeze(1)
        self.powers = (self.indices - self.indices.T).clamp(min=0)
        self.W = torch.tril(self.tau * ((1 - self.tau) ** self.powers))
        indices_1 = torch.arange(self.T, device='cuda')
        row_indices, col_indices = torch.meshgrid(indices_1, indices_1, indexing='ij')
        self.coefficients = (self.tau ** (row_indices - col_indices)).tril()
        self.coefficients2 =  self.coefficients.unsqueeze(-1).expand(-1, -1, dim)
        self.ones =  torch.ones(1, dim, device='cuda')
        self.i_indices = torch.arange(self.T, device='cuda')
        self.j_indices = torch.arange(self.T, device='cuda')
        self.i_grid, self.j_grid = torch.meshgrid(self.i_indices, self.j_indices, indexing='ij')
        self.mask = ((self.i_grid > self.j_grid) & (self.j_grid <= self.i_grid - 2)) | (self.i_grid == self.j_grid + 1)
        self.mask2 = self.mask.unsqueeze(-1)
        self.zeros = torch.zeros((1, dim), device='cuda')


    def forward(self, x: torch.Tensor):
        x_flat = x.view(self.T, -1)
        p1 = F.sigmoid(self.threshold - 0.5 * (self.tau * x_flat + torch.matmul(self.W, x_flat)))
        log_cumsum = torch.cat([
            self.zeros,
            torch.cumsum(torch.log(torch.clamp(p1, min=1e-12)), dim=0)
        ])
        log_ratio = log_cumsum[self.i_grid] - log_cumsum[self.j_grid]
        log_reset = log_ratio * self.mask2.float()
        reset1 = torch.exp(log_reset)
        s = self.surrogate_function(torch.einsum('ijc,jc->ic', self.tau * reset1 * self.coefficients2, x_flat)-self.threshold)
        s = s.reshape(x.shape)
        # version 1
        # mem_min = self.tau * x
        # indices = torch.arange(self.T, device=x.device).unsqueeze(1) 
        # powers = (indices - indices.T).clamp(min=0) 
        # W = torch.tril(self.tau * ((1-self.tau) ** powers)) 
        # x_flat = x.view(self.T, -1)
        # mem_max_flat = torch.matmul(W, x_flat)
        # mem_max = mem_max_flat.view_as(x)
        # sig_x1 = F.sigmoid(0.5 * (mem_min + mem_max) - self.threshold)
        # indices_1 = torch.arange(self.T, device=x.device)  # [0, 1, ..., T-1]
        # row_indices, col_indices = torch.meshgrid(indices_1, indices_1, indexing='ij')
        # coefficients = (self.tau ** (row_indices - col_indices)).tril()
        # reset1 = torch.ones((self.T, self.T, x_flat.shape[1]), device=x.device)
        # p1 = (1 - sig_x1).view(self.T, -1)
        # for i in range(self.T):
        #     for j in range(i):  # 确保 j <= i
        #         reset1[i, j] = torch.prod(p1[j:i], dim=0)
        # mem_coeff1 = self.tau * reset1 * (coefficients.unsqueeze(-1).expand(-1, -1, x_flat.shape[1]))
        # h1 = torch.bmm(mem_coeff1.permute(2, 0, 1), x.reshape(self.T, 1, -1).permute(2, 0, 1))
        # h1 = h1.squeeze(2).permute(1, 0)
        # h = (h1.flatten(0, 1)).view_as(x)
        # s = self.surrogate_function(h-self.threshold)
        return s