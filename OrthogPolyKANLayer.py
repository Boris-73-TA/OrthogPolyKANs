import torch
import torch.nn as nn
from BesselKANLayer import BesselKANLayer
from Cheby2KANLayer import Cheby2KANLayer
from FibonacciKANLayer import FibonacciKANLayer
from LucasKANLayer import LucasKANLayer
from LegendreKANLayer import LegendreKANLayer
from HermiteKANLayer import HermiteKANLayer
from GegenbauerKANLayer import GegenbauerKANLayer
from JacobiKANLayer import JacobiKANLayer

# Bessel, Cheby2, Fibonacci, Lucas, Legendre, Laguerre, Hermite, Gegenbauer,  Jacobi

class OrthogPolyKANLayer(nn.Module):
    def __init__(self, poly_type, input_dim, output_dim, degree, alpha=None, beta=None):
        super(OrthogPolyKANLayer, self).__init__()
        self.poly_type = poly_type.lower()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.alpha = alpha
        self.beta = beta

        # Initialize the appropriate layer based on the polynomial type
        if self.poly_type == 'jacobi':
            self.layer = JacobiKANLayer(input_dim, output_dim, degree, alpha, beta)
        elif self.poly_type == 'gegenbauer':
            self.layer = GegenbauerKANLayer(input_dim, output_dim, degree, alpha)
        elif self.poly_type == 'hermite':
            self.layer = HermiteKANLayer(input_dim, output_dim, degree)
        elif self.poly_type == 'lucas':
            self.layer = LucasKANLayer(input_dim, output_dim, degree)
        elif self.poly_type == 'fibonacci':
            self.layer = FibonacciKANLayer(input_dim, output_dim, degree)
        elif self.poly_type == 'legendre':
            self.layer = LegendreKANLayer(input_dim, output_dim, degree)
        elif self.poly_type == 'bessel':
            self.layer = BesselKANLayer(input_dim, output_dim, degree)
        elif self.poly_type == 'Cheby2':
            self.layer = Cheby2KANLayer(input_dim, output_dim, degree)
        elif self.poly_type == 'laguerre':
            self.layer = LaguerreKANLayer(input_dim, output_dim, degree, alpha)  
        else:
            raise ValueError(f"Unsupported polynomial type: {self.poly_type}, please select one of the following: Bessel, Cheby2, Fibonacci, Lucas, Legendre, Laguerre, Hermite, Gegenbauer,  Jacobi. ")

    def forward(self, x):
        return self.layer(x)
