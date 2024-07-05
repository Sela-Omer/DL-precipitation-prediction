import torch
from torch import nn


class PolynomialPredictor(nn.Module):
    def __init__(self, degree):
        super(PolynomialPredictor, self).__init__()
        self.degree = degree

    def forward(self, coefficients, last_n_timesteps):
        """
        Forward pass for predicting the next value based on polynomial coefficients and last n timesteps.

        Args:
        coefficients (torch.Tensor): Coefficients tensor of shape (batch_size, (degree + 1) * (num_timesteps))
        last_n_timesteps (torch.Tensor): Last n timesteps tensor of shape (batch_size, num_timesteps)

        Returns:
        torch.Tensor: Predicted value tensor of shape (batch_size, 1)
        """
        coefficients = coefficients.reshape(coefficients.shape[0], self.degree + 1, last_n_timesteps.size(1))
        batch_size, _, num_timesteps = coefficients.size()  # coefficients shape: (batch_size, degree + 1, num_timesteps)
        powers = [last_n_timesteps ** i for i in
                  range(self.degree + 1)]  # list of tensors each of shape (batch_size, num_timesteps)
        powers = torch.stack(powers, dim=1)  # stack into a tensor of shape (batch_size, degree + 1, num_timesteps)

        # Calculate polynomial value by summing the product of coefficients and powers for each timestep
        prediction = torch.sum(coefficients * powers, dim=1).sum(dim=1, keepdim=True)  # shape: (batch_size, 1)
        return prediction
