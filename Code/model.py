import torch
import torch.nn as nn
import layer

""" B: batch size; T: sequence length; H: hidden size."""

class LSTM(nn.Module):
    """Real-valued LSTM"""

    def __init__(self, input_dim, hidden_dim, return_sequences=False):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = layer.LSTMCell(input_dim, hidden_dim)
        self.return_sequences = return_sequences

    def forward(self, inputs, initial_state=None):
        """Input: [B, T, input_dim]; return [B, T, H] or [B, H]. Optional state: (h0, c0)."""
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        h, c = (torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)) if initial_state is None else initial_state

        # Process the input window sequentially and collect hidden states.
        outputs = []
        for t in range(seq_len):
            input_t = inputs[:, t, :]
            h, c = self.cell(input_t, (h, c))
            outputs.append(h)
        outputs = torch.stack(outputs, dim=1)
        return outputs if self.return_sequences else outputs[:, -1, :]

class ComplexMultiLayerLSTM(nn.Module):
    """Stacked complex-valued LSTM"""

    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1, return_sequences=False):
        super(ComplexMultiLayerLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences
        self.dropout = dropout

        # Each layer maintains separate real and imaginary hidden/cell states.
        self.cells = nn.ModuleList([layer.ComplexLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])

        # Apply dropout between layers only; a single layer uses no inter-layer dropout.
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers - 1)])

    def forward(self, xr, xi):
        """Inputs xr, xi: [B, T, input_dim]; return a real/imaginary pair [B, T, H] or [B, H]."""
        batch_size, seq_len, _ = xr.shape
        device = xr.device

        # Initialize the real/imaginary states of every layer to zero.
        h = [(torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)) for _ in range(self.num_layers)]
        c = [(torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)) for _ in range(self.num_layers)]
        outputs_r, outputs_i = ([], [])
        for t in range(seq_len):
            xrt, xit = (xr[:, t, :], xi[:, t, :])
            for i, cell in enumerate(self.cells):
                (hr, hi), (cr, ci) = cell(xrt, xit, (h[i], c[i]))
                h[i] = (hr, hi)
                c[i] = (cr, ci)
                xrt, xit = (hr, hi)
                if i < self.num_layers - 1:
                    xrt = self.dropouts[i](xrt)
                    xit = self.dropouts[i](xit)

            # Collect only the final layer output at each time step.
            outputs_r.append(h[-1][0])
            outputs_i.append(h[-1][1])
        outputs_r = torch.stack(outputs_r, dim=1)
        outputs_i = torch.stack(outputs_i, dim=1)
        if self.return_sequences:
            return outputs_r, outputs_i
        else:
            return (outputs_r[:, -1, :], outputs_i[:, -1, :])

class ReLSTM(nn.Module):
    """Relation LSTM"""

    def __init__(self, input_dim, cor_dim, hidden_dim, return_sequences=False):
        super(ReLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences
        self.cell = layer.LSTM_c_torch(input_dim=input_dim, cor_dim=cor_dim, hidden_dim=hidden_dim)

    def forward(self, inputs, cors, initial_state=None):
        """Inputs: [B, T, input_dim], relations: [B, T, cor_dim]; return [B, T, H] or [B, H]."""
        B, T, _ = inputs.shape
        if initial_state is None:
            h = inputs.new_zeros(B, self.hidden_dim)
            c = inputs.new_zeros(B, self.hidden_dim)
        else:
            h, c = initial_state

        # Precompute input and relation projections for the whole sequence.
        x_proj = torch.matmul(inputs, self.cell.W_all)
        cor_proj = torch.sigmoid(torch.matmul(cors, self.cell.W_cor) + self.cell.b_cor)
        cor_gate_proj = torch.matmul(cor_proj, self.cell.U_all)

        # Collect all hidden states only when sequence output is requested.
        if self.return_sequences:
            outputs = []
            for t in range(T):
                gates = x_proj[:, t, :] + cor_gate_proj[:, t, :] + torch.matmul(h, self.cell.H_all) + self.cell.b_all
                i_gate, f_gate, o_gate, c_gate = gates.chunk(4, dim=-1)
                i = torch.sigmoid(i_gate)
                f = torch.sigmoid(f_gate)
                o = torch.sigmoid(o_gate)
                c_hat = torch.tanh(c_gate)
                c = f * c + i * c_hat
                h = o * torch.tanh(c)
                outputs.append(h)
            return torch.stack(outputs, dim=1)
        for t in range(T):
            gates = x_proj[:, t, :] + cor_gate_proj[:, t, :] + torch.matmul(h, self.cell.H_all) + self.cell.b_all
            i_gate, f_gate, o_gate, c_gate = gates.chunk(4, dim=-1)
            i = torch.sigmoid(i_gate)
            f = torch.sigmoid(f_gate)
            o = torch.sigmoid(o_gate)
            c_hat = torch.tanh(c_gate)
            c = f * c + i * c_hat
            h = o * torch.tanh(c)
        return h

class RNN(nn.Module):
    """Baseline with independent real and imaginary recurrent branches."""

    def __init__(self, window_size, i_x):
        super(RNN, self).__init__()
        self.i_x = i_x
        self.window_size = window_size

        # Encode real and imaginary inputs independently.
        self.lstm_real = LSTM(input_dim=i_x, hidden_dim=i_x, return_sequences=False)
        self.lstm_imag = LSTM(input_dim=i_x, hidden_dim=i_x, return_sequences=False)

        # Map final hidden states to normalized field components in [0, 1].
        self.real_fc = nn.Sequential(nn.Linear(i_x, i_x), nn.ReLU(), nn.Linear(i_x, i_x), nn.Sigmoid())
        self.imag_fc = nn.Sequential(nn.Linear(i_x, i_x), nn.ReLU(), nn.Linear(i_x, i_x), nn.Sigmoid())

    def forward(self, real_input, imag_input):
        """Inputs: real/imaginary [B, T, i_x]; return two tensors [B, 1, i_x]."""
        x_out = self.lstm_real(real_input)
        y_out = self.lstm_imag(imag_input)
        real_output = self.real_fc(x_out)
        imag_output = self.imag_fc(y_out)

        # Restore the single prediction-step dimension: [B, i_x] -> [B, 1, i_x].
        real_output = real_output.unsqueeze(1)
        imag_output = imag_output.unsqueeze(1)
        return real_output, imag_output

class CVNN(nn.Module):
    """Baseline with a coupled complex-valued recurrent encoder."""

    def __init__(self, window_size, i_x):
        super(CVNN, self).__init__()
        self.i_x = i_x
        self.window_size = window_size

        # Complex recurrent projections couple the real and imaginary components.
        self.lstm = ComplexMultiLayerLSTM(input_dim=i_x, hidden_dim=i_x, num_layers=1, dropout=0.1, return_sequences=False)

        # Map final hidden states to normalized field components in [0, 1].
        self.real_fc = nn.Sequential(nn.Linear(i_x, i_x), nn.ReLU(), nn.Linear(i_x, i_x), nn.Sigmoid())
        self.imag_fc = nn.Sequential(nn.Linear(i_x, i_x), nn.ReLU(), nn.Linear(i_x, i_x), nn.Sigmoid())

    def forward(self, real_input, imag_input):
        """Inputs: real/imaginary [B, T, i_x]; return two tensors [B, 1, i_x]."""
        B, T, i_x = real_input.shape
        x_out, y_out = self.lstm(real_input, imag_input)
        real_output = self.real_fc(x_out)
        imag_output = self.imag_fc(y_out)

        # Restore the single prediction-step dimension: [B, i_x] -> [B, 1, i_x].
        real_output = real_output.unsqueeze(1)
        imag_output = imag_output.unsqueeze(1)
        return real_output, imag_output

class CNP(nn.Module):
    """Coherent neural propagation"""

    def __init__(self, window_size, i_x):
        super(CNP, self).__init__()
        self.i_x = i_x
        self.window_size = window_size
        self.use_residual = True

        # Extract local relations across time steps and field features.
        self.cnn = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 127), stride=(1, 1), padding=(1, 63))

        # ReLSTM branches.
        self.lstm_real = ReLSTM(input_dim=i_x, cor_dim=i_x, hidden_dim=512, return_sequences=True)
        self.lstm_imag = ReLSTM(input_dim=i_x, cor_dim=i_x, hidden_dim=512, return_sequences=True)

        # Regularize recurrent inputs during training; the relation branch uses original inputs.
        self.input_real_dropout = nn.Dropout(0.15)
        self.input_imag_dropout = nn.Dropout(0.15)

        # Mix the two branches and aggregate the full input window.
        self.time_conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(window_size, 1), stride=(1, 1), padding=(0, 0))

        # Jointly project both branches to the real and imaginary field features.
        self.feature_conv = nn.Conv2d(in_channels=512, out_channels=2 * i_x, kernel_size=(2, 1), stride=(1, 1), padding=(0, 0))
        if self.use_residual:
            # Initialize separate learnable residual blending weights.
            init_alpha = 0.15
            init_logit = torch.logit(torch.tensor(init_alpha))
            self.res_scale_real = nn.Parameter(init_logit.clone())
            self.res_scale_imag = nn.Parameter(init_logit.clone())

    def forward(self, real_input, imag_input):
        """Inputs: real/imaginary [B, T, i_x]; return two tensors [B, 1, i_x]."""
        B, T, i_x = real_input.shape
        real = self.input_real_dropout(real_input)
        imag = self.input_imag_dropout(imag_input)

        # Build the shared relation signal: [B, 2, T, i_x] -> [B, T, i_x].
        x = torch.stack([real_input, imag_input], dim=1)
        relation = self.cnn(x).squeeze(1)

        # Encode each field component with ReLSTM: [B, T, 512].
        x_out = self.lstm_real(real, relation)
        y_out = self.lstm_imag(imag, relation)

        # Aggregate time, then project features: [B, 2, T, 512] -> [B, 2*i_x, 1, 1].
        z = torch.stack([x_out, y_out], dim=1)
        z = self.time_conv(z)
        z = z.permute(0, 3, 1, 2)
        z = self.feature_conv(z)

        # Split real/imaginary features and reshape each output to [B, 1, i_x].
        z = z.view(B, 2, self.i_x, 1)
        real_output = z[:, 0, :, :]
        imag_output = z[:, 1, :, :]
        real_output = real_output.permute(0, 2, 1)
        imag_output = imag_output.permute(0, 2, 1)
        if self.use_residual:
            # Blend with the last input: (1 - alpha) * prediction + alpha * last_input.
            residual_real = real_input[:, -1:, :]
            residual_imag = imag_input[:, -1:, :]
            alpha_real = torch.sigmoid(self.res_scale_real)
            alpha_imag = torch.sigmoid(self.res_scale_imag)
            real_output = real_output + alpha_real * (residual_real - real_output)
            imag_output = imag_output + alpha_imag * (residual_imag - imag_output)

        # Optional rollout calibration scales the predicted change; missing gain defaults to 1.
        gain = getattr(self, 'delta_gain', 1.0)
        if gain != 1.0:
            real_output = real_input[:, -1:, :] + gain * (real_output - real_input[:, -1:, :])
            imag_output = imag_input[:, -1:, :] + gain * (imag_output - imag_input[:, -1:, :])
        return real_output, imag_output
