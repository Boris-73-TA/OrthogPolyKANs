import torch
import torch.nn as nn
import numpy as np

class GegenbauerKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, alpha_param):
        super(GegenbauerKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.alpha_param = alpha_param

        # Initialize Gegenbauer polynomial coefficients
        self.gegenbauer_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.gegenbauer_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Reshape to (batch_size, input_dim)
        x = torch.tanh(x)  # Normalize x to [-1, 1]
        
        gegenbauer = torch.ones(x.shape[0], self.input_dim, self.degree + 1, device=x.device)
        if self.degree > 0:
            gegenbauer[:, :, 1] = 2 * self.alpha_param * x  # C_1^alpha(x) = 2*alpha*x

        for n in range(1, self.degree):
            term1 = 2 * (n + self.alpha_param) * x * gegenbauer[:, :, n].clone()
            term2 = (n + 2 * self.alpha_param - 1) * gegenbauer[:, :, n - 1].clone()
            gegenbauer[:, :, n + 1] = (term1 - term2) / (n + 1)  # Apply the recurrence relation

        y = torch.einsum('bid,iod->bo', gegenbauer, self.gegenbauer_coeffs)
        return y.view(-1, self.output_dim)

