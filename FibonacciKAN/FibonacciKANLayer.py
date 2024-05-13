import torch
import torch.nn as nn
import numpy as np

class FibonacciKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(FibonacciKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        # Initialize coefficients for the Fibonacci polynomials
        self.fib_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.fib_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Reshape to (batch_size, input_dim)
        x = torch.tanh(x)  # Normalize input x to [-1, 1] for stability in polynomial calculation

        # Initialize Fibonacci polynomial tensors
        fib = torch.zeros(x.size(0), self.input_dim, self.degree + 1, device=x.device)
        fib[:, :, 0] = 0  # F_0(x) = 0
        if self.degree > 0:
            fib[:, :, 1] = 1  # F_1(x) = 1

        for i in range(2, self.degree + 1):
            # Compute Fibonacci polynomials using the recurrence relation
            fib[:, :, i] = x * fib[:, :, i - 1].clone() + fib[:, :, i - 2].clone()

        # Normalize the polynomial outputs to prevent runaway values
        #fib = torch.tanh(fib)

        # Compute the Fibonacci interpolation
        y = torch.einsum('bid,iod->bo', fib, self.fib_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)
        return y
        
# Add orthogonalization ?