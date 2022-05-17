from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNNPolicy(nn.Module):
    """A policy function parametrized by a fully-connected neural network.
    The model architecture uses fully-connected layers.
    After applying the non-linearity a dropout layer is applied.
    For a network with L layers the architecture will be:
    {affine - leaky-ReLU - [dropout]} x (L - 1) - affine
    The weights of the layers are initialized using Kaiming He uniform distribution.
    Attributes:
        input_size (int): The size of the input to the network
        hidden_sizes (list(int)): A list of sizes of the hidden layers.
        out_size (int): The size of the output layer of the network.
        dropout_rate (float): Dropout probability.
    """

    def __init__(self, input_shape, hidden_sizes, out_size, dropout_rate=0.0):
        """Initialize a policy model.
        Args:
            input_shape (tuple[int]): The shape of the environment observable state.
            hidden_sizes (list[int]): A list of sizes for the hidden layers. Providing an
                empty slice will create a linear function approximation with no hidden layers.
            out_size (int): Number of possible actions the agent can choose from.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        self.input_size = reduce(lambda x, y: x * y, input_shape)
        self.hidden_sizes = hidden_sizes
        self.out_size = out_size
        self.dropout_rate = dropout_rate

        # Store arguments for model initialization.
        # Kwargs dict is used to save and restore the model.
        self.kwargs = dict(input_shape=input_shape,
                           hidden_sizes=self.hidden_sizes,
                           out_size=self.out_size,
                           dropout_rate=self.dropout_rate)

        # Initialize the model architecture.
        self.num_layers = len(hidden_sizes)
        self.hidden_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        fan_in = self.input_size
        fan_out = self.input_size
        for fan_out in hidden_sizes:
            self.hidden_layers.append(nn.Linear(fan_in, fan_out))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            fan_in = fan_out
        self.output_layer = nn.Linear(fan_out, out_size, bias=False)

        # Initialize model parameters.
        for param in self.parameters():
            if len(param.shape) >= 2:
                nn.init.kaiming_uniform_(param)         # weight
            else:
                nn.init.uniform_(param, -0.01, 0.01)    # bias


    def forward(self, x):
        """Take a mini-batch of environment states and compute scores over the possible
        actions.
        Args:
            x (torch.Tensor): Tensor of shape (b, q), or (b, t, q), giving the current
                state of the environment, where b = batch size, t = number of time steps,
                q = size of the quantum system (2 ** num_qubits).

        Returns:
            out (torch.Tensor): Tensor of shape (b, num_actions), or (b, t, num_acts),
                giving a score to every action from the action space.
        """
        out = x
        for idx in range(self.num_layers):
            out = self.hidden_layers[idx](out)
            out = F.leaky_relu(out)
            out = self.dropout_layers[idx](out)
        out = self.output_layer(out)
        return out

    @property
    def device(self):
        """str: Determine which device to place the Tensors upon, CPU or GPU."""
        return self.output_layer.weight.device

    @classmethod
    def load(cls, model_path):
        """Load the model from a file."""
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        kwargs = params["kwargs"]
        model = cls(**kwargs)
        model.load_state_dict(params["state_dict"])
        return model

    def save(self, path):
        """Save the model to a file."""
        params = {"kwargs": self.kwargs,
                  "state_dict": self.state_dict()
                 }
        torch.save(params, path)

#