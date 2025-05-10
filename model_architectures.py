import torch
import torch.nn as nn
import snntorch as snn
import torch_geometric as pyg
from torch_geometric.nn import GCNConv
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, input_projection_size, hidden_size, dense_size, dropout):
        super(LSTM, self).__init__()
        self.input_projection = nn.Linear(input_size, input_projection_size)
        self.lstm = nn.LSTM(input_size=input_projection_size, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, dense_size)
        self.output = nn.Linear(dense_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_projection(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.dropout(lstm_out)
        out = torch.relu(self.dense(out))
        out = self.output(out)
        out = self.sigmoid(out)
        return out

class CfcCell(nn.Module):
    def __init__(self, input_size, hidden_size, hparams):
        super(CfcCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hparams = hparams
        self._no_gate = hparams.get("no_gate", False)
        self._minimal = hparams.get("minimal", False)

        if hparams["backbone_activation"] == "silu":
            backbone_activation = nn.SiLU
        elif hparams["backbone_activation"] == "relu":
            backbone_activation = nn.ReLU
        elif hparams["backbone_activation"] == "tanh":
            backbone_activation = nn.Tanh
        elif hparams["backbone_activation"] == "gelu":
            backbone_activation = nn.GELU
        else:
            raise ValueError("Unknown activation")

        layer_list = [
            nn.Linear(input_size + hidden_size, hparams["backbone_units"]),
            backbone_activation(),
        ]
        for i in range(1, hparams["backbone_layers"]):
            layer_list.append(nn.Linear(hparams["backbone_units"], hparams["backbone_units"]))
            layer_list.append(backbone_activation())
            if "backbone_dr" in hparams:
                layer_list.append(nn.Dropout(hparams["backbone_dr"]))
        self.backbone = nn.Sequential(*layer_list)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.ff1 = nn.Linear(hparams["backbone_units"], hidden_size)
        
        if self._minimal:
            self.w_tau = nn.Parameter(torch.zeros(1, hidden_size))
            self.A = nn.Parameter(torch.ones(1, hidden_size))
        else:
            self.ff2 = nn.Linear(hparams["backbone_units"], hidden_size)
            self.time_a = nn.Linear(hparams["backbone_units"], hidden_size)
            self.time_b = nn.Linear(hparams["backbone_units"], hidden_size)
        
        self.init_weights()

    def init_weights(self):
        init_gain = self.hparams.get("init")
        if init_gain is not None:
            for w in self.parameters():
                if w.dim() == 2:
                    torch.nn.init.xavier_uniform_(w, gain=init_gain)

    def forward(self, input, hx, ts):
        batch_size = input.size(0)
        ts = ts.view(batch_size, 1)
        x = torch.cat([input, hx], 1)
        x = self.backbone(x)
        
        if self._minimal:
            ff1 = self.ff1(x)
            new_hidden = (-self.A * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1))) * ff1 + self.A)
        else:
            ff1 = self.tanh(self.ff1(x))
            ff2 = self.tanh(self.ff2(x))
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = self.sigmoid(t_a * ts + t_b)
            new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
        return new_hidden

class Cfc(nn.Module):
    def __init__(self, in_features, hidden_size, out_feature, hparams, return_sequences=False):
        super(Cfc, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_feature = out_feature
        self.return_sequences = return_sequences
        self.rnn_cell = CfcCell(in_features, hidden_size, hparams)
        self.fc = nn.Linear(hidden_size, out_feature)

    def forward(self, x, timespans=None):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        h_state = torch.zeros((batch_size, self.hidden_size), device=device)
        
        if timespans is None:
            timespans = torch.ones(batch_size, seq_len, device=device)
        
        if self.return_sequences:
            output_sequence = []
            for t in range(seq_len):
                inputs = x[:, t]
                ts = timespans[:, t]
                h_state = self.rnn_cell(inputs, h_state, ts)
                output_sequence.append(self.fc(h_state))
            return torch.stack(output_sequence, dim=1)
        else:
            for t in range(seq_len):
                inputs = x[:, t]
                ts = timespans[:, t]
                h_state = self.rnn_cell(inputs, h_state, ts)
            return self.fc(h_state)

class CfcClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hparams, dropout, input_projection_size, dense_size):
        super(CfcClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(input_size, input_projection_size)
        self.cfc = Cfc(input_projection_size, hidden_size, output_size, hparams, return_sequences=False)
        self.dense = nn.Linear(output_size, dense_size)
        self.output = nn.Linear(dense_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_projection(x)
        out = self.cfc(x)
        out = torch.relu(self.dense(out))
        out = self.dropout(out)
        out = self.output(out)
        out = self.sigmoid(out)
        return out

class SnnClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hparams, dropout, input_projection_size, dense_size):
        super(SnnClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_steps = hparams.get("num_steps", 10)
        self.beta = hparams.get("beta", 0.95)
        self.threshold = hparams.get("threshold", 1.0)
        self.input_projection = nn.Linear(input_size, input_projection_size)
        self.fc1 = nn.Linear(input_projection_size, hidden_size)
        self.lif1 = snn.Leaky(beta=self.beta, threshold=self.threshold, spike_grad=None)
        self.dense = nn.Linear(hidden_size, dense_size)
        self.output = nn.Linear(dense_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_projection(x)
        mem1 = self.lif1.init_leaky()
        spk_out = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for step in range(self.num_steps):
            cur_input = x[:, step, :]
            cur1 = self.fc1(cur_input)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk_out = mem1
        out = torch.relu(self.dense(spk_out))
        out = self.dropout(out)
        out = self.output(out)
        out = self.sigmoid(out)
        return out

class LsmClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hparams, dropout, input_projection_size, dense_size):
        super(LsmClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_steps = hparams.get("num_steps", 10)
        self.beta = hparams.get("beta", 0.95)
        self.threshold = hparams.get("threshold", 1.0)
        self.spectral_radius = hparams.get("spectral_radius", 0.9)
        self.sparsity = hparams.get("sparsity", 0.1)
        self.input_projection = nn.Linear(input_size, input_projection_size)
        self.input_to_reservoir = nn.Linear(input_projection_size, hidden_size)
        reservoir_weights = torch.randn(self.hidden_size, self.hidden_size)
        mask = (torch.rand(self.hidden_size, self.hidden_size) < self.sparsity).float()
        reservoir_weights = reservoir_weights * mask
        eigenvalues = torch.linalg.eigvals(reservoir_weights).abs()
        max_eigenvalue = eigenvalues.max()
        if max_eigenvalue > 0:
            reservoir_weights = reservoir_weights * (self.spectral_radius / max_eigenvalue)
        self.reservoir_weights = nn.Parameter(reservoir_weights, requires_grad=False)
        self.lif_reservoir = snn.Leaky(beta=self.beta, threshold=self.threshold, spike_grad=None)
        self.dense = nn.Linear(hidden_size, dense_size)
        self.output = nn.Linear(dense_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_projection(x)
        reservoir_state = torch.zeros(batch_size, self.hidden_size, device=x.device)
        mem = self.lif_reservoir.init_leaky()
        for step in range(self.num_steps):
            cur_input = x[:, step, :]
            input_contrib = self.input_to_reservoir(cur_input)
            recurrent_contrib = torch.matmul(reservoir_state, self.reservoir_weights)
            reservoir_input = input_contrib + recurrent_contrib
            spk, mem = self.lif_reservoir(reservoir_input, mem)
            reservoir_state = mem
        out = torch.relu(self.dense(reservoir_state))
        out = self.dropout(out)
        out = self.output(out)
        out = self.sigmoid(out)
        return out

class EsnClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hparams, dropout, input_projection_size, dense_size):
        super(EsnClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_steps = hparams.get("num_steps", 10)
        self.spectral_radius = hparams.get("spectral_radius", 0.9)
        self.sparsity = hparams.get("sparsity", 0.1)
        self.leaking_rate = hparams.get("leaking_rate", 0.5)
        self.input_scaling = hparams.get("input_scaling", 1.0)
        self.input_projection = nn.Linear(input_size, input_projection_size)
        self.input_to_reservoir = nn.Linear(input_projection_size, hidden_size)
        with torch.no_grad():
            self.input_to_reservoir.weight.mul_(self.input_scaling)
        reservoir_weights = torch.randn(self.hidden_size, self.hidden_size)
        mask = (torch.rand(self.hidden_size, self.hidden_size) < self.sparsity).float()
        reservoir_weights = reservoir_weights * mask
        eigenvalues = torch.linalg.eigvals(reservoir_weights).abs()
        max_eigenvalue = eigenvalues.max()
        if max_eigenvalue > 0:
            reservoir_weights = reservoir_weights * (self.spectral_radius / max_eigenvalue)
        self.reservoir_weights = nn.Parameter(reservoir_weights, requires_grad=False)
        self.tanh = nn.Tanh()
        self.dense = nn.Linear(hidden_size, dense_size)
        self.output = nn.Linear(dense_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_projection(x)
        reservoir_state = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for step in range(self.num_steps):
            cur_input = x[:, step, :]
            input_contrib = self.input_to_reservoir(cur_input)
            recurrent_contrib = torch.matmul(reservoir_state, self.reservoir_weights)
            pre_activation = input_contrib + recurrent_contrib
            new_state = self.tanh(pre_activation)
            reservoir_state = (1 - self.leaking_rate) * reservoir_state + self.leaking_rate * new_state
        out = torch.relu(self.dense(reservoir_state))
        out = self.dropout(out)
        out = self.output(out)
        out = self.sigmoid(out)
        return out

class SpikingGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, beta, threshold):
        super(SpikingGCNConv, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.lif = snn.Leaky(beta=beta, threshold=threshold, spike_grad=None)

    def forward(self, x, edge_index, mem):
        x = self.conv(x, edge_index)
        spk, mem = self.lif(x, mem)
        return spk, mem

class SpikingGNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hparams, dropout, input_projection_size, dense_size):
        super(SpikingGNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_steps = hparams.get("num_steps", 10)
        self.beta = hparams.get("beta", 0.95)
        self.threshold = hparams.get("threshold", 1.0)
        self.num_nodes = hparams.get("num_nodes", 10)

        self.input_projection = nn.Linear(input_size, input_projection_size)
        self.spiking_gcn = SpikingGCNConv(input_projection_size, hidden_size, self.beta, self.threshold)
        self.dense = nn.Linear(hidden_size * self.num_nodes, dense_size)
        self.output = nn.Linear(dense_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = self.input_projection(x)
        x = x.view(batch_size * seq_len, -1)
        edge_index = []
        for b in range(batch_size):
            for t in range(seq_len - 1):
                edge_index.append([b * seq_len + t, b * seq_len + t + 1])
                edge_index.append([b * seq_len + t + 1, b * seq_len + t])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(x.device)
        mem = self.spiking_gcn.lif.init_leaky()
        for step in range(self.num_steps):
            spk, mem = self.spiking_gcn(x, edge_index, mem)
        out = mem.view(batch_size, -1)
        out = torch.relu(self.dense(out))
        out = self.dropout(out)
        out = self.output(out)
        out = self.sigmoid(out)
        return out