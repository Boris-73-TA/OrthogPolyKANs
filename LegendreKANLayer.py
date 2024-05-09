import torch
import torch.nn as nn

class LegendreKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(LegendreKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.legendre_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.legendre_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        x = torch.tanh(x)  # Normalize input to [-1, 1] for stability in Legendre polynomial calculation

        # Initialize Legendre polynomial tensors
        legendre = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        legendre[:, :, 0] = 1  # P_0(x) = 1
        if self.degree > 0:
            legendre[:, :, 1] = x  # P_1(x) = x

        # Compute Legendre polynomials using the recurrence relation
        for n in range(2, self.degree + 1):
           # Recurrence relation without in-place operations
            legendre[:, :, n] = ((2 * (n-1) + 1) / (n)) * x * legendre[:, :, n-1].clone() - ((n-1) / (n)) * legendre[:, :, n-2].clone()

        # Compute output using matrix multiplication
        y = torch.einsum('bid,iod->bo', legendre, self.legendre_coeffs)
        y = y.view(-1, self.outdim)
        return y

