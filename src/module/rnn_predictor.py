import torch
from torch import nn
import torch.nn.functional as F


class RNNPredictor(nn.Module):
    def __init__(self, hidden_size):
        super(RNNPredictor, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, coefficients, last_n_inputs):
        """
        Forward pass for predicting the next value based on RNN coefficients and last n timesteps.

        Args:
        coefficients (torch.Tensor): Coefficients tensor of shape (batch_size, hidden_size * 5 + 1 + hidden_size*hidden_size)
                                     Includes weights and biases for the RNN and fully connected layer, plus initial hidden state.
        last_n_inputs (torch.Tensor): Last n timesteps tensor of shape (batch_size, num_timesteps)

        Returns:
        torch.Tensor: Predicted value tensor of shape (batch_size, 1)
        """
        batch_size = coefficients.size(0)
        num_timesteps = last_n_inputs.size(1)

        # hidden_size * 5 + 1 + hidden_size*hidden_size
        # Define sizes for weights, biases, and initial hidden state
        weight_ih_size = self.hidden_size * 1
        weight_hh_size = self.hidden_size * self.hidden_size
        bias_ih_size = self.hidden_size
        bias_hh_size = self.hidden_size
        fc_weight_size = self.hidden_size * 1  # For final output of size 1
        fc_bias_size = 1
        h_0_size = self.hidden_size

        # Calculate the total size needed for the coefficients tensor
        total_size = (
                    weight_ih_size + weight_hh_size + bias_ih_size + bias_hh_size + fc_weight_size + fc_bias_size + h_0_size)

        # Ensure coefficients tensor size matches the needed total size
        assert coefficients.size(
            1) == total_size, "Coefficients tensor size must match the total size needed for weights, biases, and initial hidden state"

        # Extract weights, biases, and initial hidden state from coefficients
        idx = 0
        weights_ih = coefficients[:, idx:idx + weight_ih_size].view(batch_size, self.hidden_size, 1)
        idx += weight_ih_size
        weights_hh = coefficients[:, idx:idx + weight_hh_size].view(batch_size, self.hidden_size, self.hidden_size)
        idx += weight_hh_size
        bias_ih = coefficients[:, idx:idx + bias_ih_size].view(batch_size, self.hidden_size)
        idx += bias_ih_size
        bias_hh = coefficients[:, idx:idx + bias_hh_size].view(batch_size, self.hidden_size)
        idx += bias_hh_size
        fc_weights = coefficients[:, idx:idx + fc_weight_size].view(batch_size, 1, self.hidden_size)
        idx += fc_weight_size
        fc_bias = coefficients[:, idx:idx + fc_bias_size].view(batch_size, 1)
        idx += fc_bias_size
        h_0 = coefficients[:, idx:idx + h_0_size].view(1, batch_size, self.hidden_size)

        # Reshape last_n_inputs to (batch_size, num_timesteps, 1) for RNN input
        last_n_inputs = last_n_inputs.unsqueeze(-1)  # Shape: (batch_size, num_timesteps, 1)

        # Manually apply the RNN with the generated weights and biases
        hidden = h_0
        for t in range(num_timesteps):
            input_t = last_n_inputs[:, t, :]  # Shape: (batch_size, 1)
            input_t = input_t.unsqueeze(2)  # Shape: (batch_size, 1, 1)
            hidden = torch.tanh(
                torch.bmm(input_t, weights_ih.transpose(1, 2)).squeeze(2).permute(1,0,2) + bias_ih +
                torch.bmm(hidden.transpose(0, 1), weights_hh.transpose(1, 2)).squeeze(2).permute(1,0,2) + bias_hh
            )  # Shape: (1, batch_size, hidden_size)


        # Fully connected layer to produce the output using extracted weights and bias
        prediction = torch.bmm(hidden.squeeze(0).unsqueeze(1), fc_weights.permute(0, 2, 1)).squeeze(1) + fc_bias  # Shape: (batch_size, 1)

        return prediction