"""Recurrent building blocks. B: batch size; T: sequence length; H: hidden size."""
import torch
import math
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor

class LSTMCell(nn.Module):
    """Real-valued LSTM cell with packed input, forget, candidate and output gates."""

    def __init__(self, input_dim, hidden_dim, bias=True):
        super(LSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = bias

        # Pack four gate projections together to compute them in one linear operation.
        self.weight_ih = Parameter(torch.empty(4 * hidden_dim, input_dim))
        self.weight_hh = Parameter(torch.empty(4 * hidden_dim, hidden_dim))
        if bias:
            self.bias_ih = Parameter(torch.empty(4 * hidden_dim))
            self.bias_hh = Parameter(torch.empty(4 * hidden_dim))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights and biases uniformly within +/- 1/sqrt(hidden_dim)."""
        bound = 1.0 / self.hidden_dim ** 0.5
        nn.init.uniform_(self.weight_ih, -bound, bound)
        nn.init.uniform_(self.weight_hh, -bound, bound)
        if self.use_bias:
            nn.init.uniform_(self.bias_ih, -bound, bound)
            nn.init.uniform_(self.bias_hh, -bound, bound)

    def forward(self, x_t, state):
        """Input x_t: [B, input_dim]; state and output: (h, c), each [B, H]."""
        h_prev, c_prev = state

        # Combine input and recurrent projections: [B, 4*H].
        gates = torch.nn.functional.linear(x_t, self.weight_ih, self.bias_ih)
        gates = gates + torch.nn.functional.linear(h_prev, self.weight_hh, self.bias_hh)

        # Gate order matches PyTorch: input, forget, candidate, output.
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=-1)
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)

        # Update cell memory, then expose it through the output gate.
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

class LSTM_c_torch(nn.Module):
    """LSTM cell whose four gates also depend on a learned relation signal."""

    def __init__(self, input_dim, cor_dim, hidden_dim):
        super(LSTM_c_torch, self).__init__()
        self.input_dim = input_dim
        self.cor_dim = cor_dim
        self.hidden_dim = hidden_dim

        # Project the relation features to hidden size before conditioning the gates.
        self.W_cor = Parameter(Tensor(cor_dim, hidden_dim))
        self.b_cor = Parameter(Tensor(hidden_dim))

        # Packed gate order: input, forget, output, candidate.
        self.W_all = Parameter(Tensor(input_dim, 4 * hidden_dim))
        self.U_all = Parameter(Tensor(hidden_dim, 4 * hidden_dim))
        self.H_all = Parameter(Tensor(hidden_dim, 4 * hidden_dim))
        self.b_all = Parameter(Tensor(4 * hidden_dim))
        self.reset_weights()

    def reset_weights(self):
        """Initialize weights uniformly; set biases to zero."""
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.uniform_(param, -stdv, stdv)
            else:
                nn.init.zeros_(param)

    def forward(self, x_t, cor_t, state):
        """Inputs x_t: [B, input_dim], cor_t: [B, cor_dim]; state and output: (h, c), each [B, H]."""
        h_prev, c_prev = state

        # Transform the relation features with a sigmoid projection.
        cor = torch.sigmoid(torch.matmul(cor_t, self.W_cor) + self.b_cor)

        # Combine input, relation and recurrent projections for all four gates.
        gates = torch.matmul(x_t, self.W_all) + torch.matmul(cor, self.U_all) + torch.matmul(h_prev, self.H_all) + self.b_all
        i_gate, f_gate, o_gate, c_gate = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)
        c_hat = torch.tanh(c_gate)
        c_t = f * c_prev + i * c_hat
        h_t = o * torch.tanh(c_t)
        return h_t, c_t

class ComplexLSTMCell(nn.Module):
    """LSTM cell with complex affine projections and componentwise state updates."""

    def __init__(self, input_dim, hidden_dim):
        super(ComplexLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input gate parameters; suffixes r/i denote real/imaginary components.
        self.W_i_r = Parameter(Tensor(input_dim, hidden_dim))
        self.H_i_r = Parameter(Tensor(hidden_dim, hidden_dim))
        self.b_i_r = Parameter(Tensor(hidden_dim))
        self.W_i_i = Parameter(Tensor(input_dim, hidden_dim))
        self.H_i_i = Parameter(Tensor(hidden_dim, hidden_dim))
        self.b_i_i = Parameter(Tensor(hidden_dim))

        # Forget gate parameters.
        self.W_f_r = Parameter(Tensor(input_dim, hidden_dim))
        self.H_f_r = Parameter(Tensor(hidden_dim, hidden_dim))
        self.b_f_r = Parameter(Tensor(hidden_dim))
        self.W_f_i = Parameter(Tensor(input_dim, hidden_dim))
        self.H_f_i = Parameter(Tensor(hidden_dim, hidden_dim))
        self.b_f_i = Parameter(Tensor(hidden_dim))

        # Candidate cell-state parameters.
        self.W_c_r = Parameter(Tensor(input_dim, hidden_dim))
        self.H_c_r = Parameter(Tensor(hidden_dim, hidden_dim))
        self.b_c_r = Parameter(Tensor(hidden_dim))
        self.W_c_i = Parameter(Tensor(input_dim, hidden_dim))
        self.H_c_i = Parameter(Tensor(hidden_dim, hidden_dim))
        self.b_c_i = Parameter(Tensor(hidden_dim))

        # Output gate parameters.
        self.W_o_r = Parameter(Tensor(input_dim, hidden_dim))
        self.H_o_r = Parameter(Tensor(hidden_dim, hidden_dim))
        self.b_o_r = Parameter(Tensor(hidden_dim))
        self.W_o_i = Parameter(Tensor(input_dim, hidden_dim))
        self.H_o_i = Parameter(Tensor(hidden_dim, hidden_dim))
        self.b_o_i = Parameter(Tensor(hidden_dim))
        self.reset_weights()

    def reset_weights(self):
        """Initialize weights uniformly; set biases to zero."""
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for name, param in self.named_parameters():
            if name.startswith('b_'):
                nn.init.zeros_(param)
            else:
                nn.init.uniform_(param, -stdv, stdv)

    def forward(self, x_r, x_i, state):
        """Inputs: [B, input_dim]; state and output: ((h_r, h_i), (c_r, c_i)), each tensor [B, H]."""
        (h_prev_r, h_prev_i), (c_prev_r, c_prev_i) = state

        # Complex projection: real = xr @ Wr - xi @ Wi, imaginary = xi @ Wr + xr @ Wi.
        i_r = torch.sigmoid(x_r @ self.W_i_r - x_i @ self.W_i_i + h_prev_r @ self.H_i_r - h_prev_i @ self.H_i_i + self.b_i_r)
        i_i = torch.sigmoid(x_i @ self.W_i_r + x_r @ self.W_i_i + h_prev_i @ self.H_i_r + h_prev_r @ self.H_i_i + self.b_i_i)

        # Compute real and imaginary forget gates with componentwise sigmoid activations.
        f_r = torch.sigmoid(x_r @ self.W_f_r - x_i @ self.W_f_i + h_prev_r @ self.H_f_r - h_prev_i @ self.H_f_i + self.b_f_r)
        f_i = torch.sigmoid(x_i @ self.W_f_r + x_r @ self.W_f_i + h_prev_i @ self.H_f_r + h_prev_r @ self.H_f_i + self.b_f_i)

        # Compute real and imaginary output gates.
        o_r = torch.sigmoid(x_r @ self.W_o_r - x_i @ self.W_o_i + h_prev_r @ self.H_o_r - h_prev_i @ self.H_o_i + self.b_o_r)
        o_i = torch.sigmoid(x_i @ self.W_o_r + x_r @ self.W_o_i + h_prev_i @ self.H_o_r + h_prev_r @ self.H_o_i + self.b_o_i)

        # Compute the candidate memory using componentwise tanh activations.
        g_r = torch.tanh(x_r @ self.W_c_r - x_i @ self.W_c_i + h_prev_r @ self.H_c_r - h_prev_i @ self.H_c_i + self.b_c_r)
        g_i = torch.tanh(x_i @ self.W_c_r + x_r @ self.W_c_i + h_prev_i @ self.H_c_r + h_prev_r @ self.H_c_i + self.b_c_i)

        # Update real and imaginary memories componentwise (no complex multiplication here).
        c_r = f_r * c_prev_r + i_r * g_r
        c_i = f_i * c_prev_i + i_i * g_i

        # Apply the output gates to the activated memories.
        h_r = o_r * torch.tanh(c_r)
        h_i = o_i * torch.tanh(c_i)
        return ((h_r, h_i), (c_r, c_i))
