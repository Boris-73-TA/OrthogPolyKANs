import torch
import torch.nn as nn
import numpy as np

class BesselKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(BesselKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        # Initialize Bessel polynomial coefficients
        self.bessel_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.bessel_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Reshape x to (batch_size, input_dim)
        # Normalize x to [-1, 1] using tanh
        x = torch.tanh(x)

        # Initialize Bessel polynomial tensors
        bessel = torch.ones(x.shape[0], self.input_dim, self.degree + 1, device=x.device)
        if self.degree > 0:
            bessel[:, :, 1] = x + 1  # y1(x) = x + 1
        for i in range(2, self.degree + 1):
            bessel[:, :, i] = (2 * i - 1) * x * bessel[:, :, i - 1].clone() + bessel[:, :, i - 2].clone()

        # Bessel interpolation using einsum for batched matrix-vector multiplication
        y = torch.einsum('bid,iod->bo', bessel, self.bessel_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)
        return y
