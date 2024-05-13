import torch
import torch.nn as nn
import numpy as np

class LaguerreKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, alpha):
        super(LaguerreKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.alpha = alpha  # Alpha parameter for generalized Laguerre polynomials

        # Initialize coefficients for the Laguerre polynomials
        self.laguerre_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.laguerre_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Reshape to (batch_size, input_dim)
        x = torch.tanh(x)  # Normalize input x to [-1, 1] for stability in polynomial calculation

        # Initialize Laguerre polynomial tensors
        laguerre = torch.zeros(x.size(0), self.input_dim, self.degree + 1, device=x.device)
        laguerre[:, :, 0] = 1  # L_0^alpha(x) = 1
        if self.degree > 0:
            laguerre[:, :, 1] = 1 + self.alpha - x  # L_1^alpha(x) = 1 + alpha - x

        for k in range(2, self.degree + 1):
            # Compute Laguerre polynomials using the generalized recurrence relation
            term1 = ((2 * (k-1) + 1 + self.alpha - x) * laguerre[:, :, k - 1].clone())
            term2 = (k - 1 + self.alpha) * laguerre[:, :, k - 2].clone()
            laguerre[:, :, k] = (term1 - term2) / (k)

        # Normalize the polynomial outputs to prevent runaway values
        #laguerre = torch.tanh(laguerre)

        # Compute the Laguerre interpolation
        y = torch.einsum('bid,iod->bo', laguerre, self.laguerre_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)
        return y
        