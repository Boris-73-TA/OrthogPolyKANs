import torch
import torch.nn as nn
import numpy as np

class JacobiKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, alpha, beta):
        super(JacobiKANLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim
        self.degree = degree
        self.alpha = alpha
        self.beta = beta

        # Initialize Jacobi polynomial coefficients
        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.input_dim))
        x = torch.tanh(x)  # Normalize x to [-1, 1]
        jacobi = torch.ones(x.shape[0], self.input_dim, self.degree + 1, device=x.device)
        if self.degree > 0:
            jacobi[:, :, 1] = (0.5 * (self.alpha - self.beta) + (self.alpha + self.beta + 2) * x / 2)
        for n in range(2, self.degree + 1):
            A_n = 2 * n * (n + self.alpha + self.beta) * (2 * n + self.alpha + self.beta - 2)
            B_n = (2 * n + self.alpha + self.beta - 1) * (2 * n + self.alpha + self.beta) * \
                  (2 * n + self.alpha + self.beta - 2) + (self.alpha ** 2 - self.beta ** 2)
            C_n = (2 * n + self.alpha + self.beta - 1) * (n + self.alpha - 1) * (n + self.beta - 1) * \
                  (2 * n + self.alpha + self.beta)

            term1 = (2 * n + self.alpha + self.beta - 1) * (2 * n + self.alpha + self.beta) * \
                    (2 * n + self.alpha + self.beta - 2) * x * jacobi[:, :, n-1].clone()
            term2 = (2 * n + self.alpha + self.beta - 1) * (self.alpha ** 2 - self.beta ** 2) * jacobi[:, :, n-1].clone()
            term3 = (n + self.alpha + self.beta - 1) * (n + self.alpha - 1) * (n + self.beta - 1) * \
                    (2 * n + self.alpha + self.beta) * jacobi[:, :, n-2].clone()

            jacobi[:, :, n] = (term1 - term2 - term3) / A_n

        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs)
        y = y.view(-1, self.out_dim)
        return y

