import torch.nn as nn
import torch
import torch.nn.functional as F


class TabularModel(nn.Module):
    def __init__(self, start_neurons):
        super(TabularModel, self).__init__()

        # Embedding layers
        input_dims = [6, 7, 3, 13]
        self.embeddings = nn.ModuleList([nn.Embedding(dim, start_neurons) for dim in input_dims])
        self.last_dense = nn.Linear(1, start_neurons)

        total_embedding_dim = (len(input_dims) + 1) * start_neurons

        # Main layers
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(5)])
        self.gates = nn.ModuleList([nn.Linear(total_embedding_dim, total_embedding_dim) for _ in range(5)])
        self.main_denses = nn.ModuleList([nn.Linear(2 * total_embedding_dim, total_embedding_dim) for _ in range(5)])

        # Output layer
        self.output = nn.Linear(total_embedding_dim, 1)

    def forward(self, x):
        embeddings = []
        for i, e in enumerate(self.embeddings):
            embeddings.append(e(x[:, i]))

        embeddings.append(self.last_dense(x[:, -1].float().unsqueeze(-1)))

        all_layer = torch.cat(embeddings, 1)

        for i in range(5):
            all_layer_d = self.dropouts[i](all_layer)
            all_layer_d_gate = torch.sigmoid(self.gates[i](all_layer_d))
            all_layer_ = all_layer * all_layer_d_gate
            all_layer_c = torch.cat([all_layer, all_layer_], 1)
            all_layer += F.relu(self.main_denses[i](all_layer_c))

        output = self.output(all_layer).squeeze(-1)

        return output


class TabularModel1(nn.Module):
    def __init__(self, start_neurons):
        super(TabularModel1, self).__init__()

        self.input_dims = [6, 7, 3, 13]

        self.embeddings = nn.ModuleList([nn.Embedding(dim, start_neurons) for dim in self.input_dims])
        self.last_dense = nn.Linear(1, start_neurons)

        self.dropout = nn.Dropout(0.2)
        self.middle_layers = nn.Sequential(
            nn.Linear(start_neurons * (len(self.input_dims)), start_neurons * (len(self.input_dims))),
            nn.Sigmoid(),
            nn.Linear(start_neurons * (len(self.input_dims)) * 2, start_neurons * ((len(self.input_dims)) + 1))
        )

        self.output_layer = nn.Linear(start_neurons * ((len(self.input_dims)) + 1), 1)

    def forward(self, x):
        embeddings = [embedding(x[:, i]) for i, embedding in enumerate(self.embeddings)]
        last_dense_out = self.last_dense(x[:, -1])
        concatenated = torch.cat(embeddings + [last_dense_out], dim=1)

        all_layer = concatenated

        for _ in range(5):
            all_layer_d = self.dropout(all_layer)
            all_layer_d_gate = self.middle_layers[0](all_layer_d)
            all_layer_ = all_layer * all_layer_d_gate
            all_layer_c = torch.cat([all_layer, all_layer_], dim=1)
            all_layer += self.middle_layers[2](all_layer_c)

        outputs = [self.output_layer(all_layer) for _ in range(10)]
        final_output = torch.mean(torch.stack(outputs, dim=1), dim=1)

        return final_output


class TabularModel(nn.Module):
    def __init__(self, input_dims, start_neurons):
        super(TabularModel, self).__init__()

        # Embedding layers
        self.embeddings = nn.ModuleList([nn.Embedding(dim, start_neurons) for dim in input_dims[:-1]])
        self.linear_embedding = nn.Linear(1, start_neurons)

        # Main layers
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(5)])
        self.gates = nn.ModuleList(
            [nn.Linear(start_neurons * len(input_dims), start_neurons * len(input_dims)) for _ in range(5)])
        self.linear_layers = nn.ModuleList(
            [nn.Linear(start_neurons * len(input_dims) * 2, start_neurons * len(input_dims)) for _ in range(5)])

        # Output layers
        self.output_layers = nn.ModuleList([nn.Linear(start_neurons * len(input_dims), 1) for _ in range(10)])

    def forward(self, x):
        embeddings = [self.embeddings[i](x[:, i]) for i in range(len(input_dims[:-1]))]
        embeddings.append(self.linear_embedding(x[:, -1].float().unsqueeze(-1)))

        concatenated_inputs = torch.cat(embeddings, dim=1)

        for dropout, gate, dense in zip(self.dropouts, self.gates, self.linear_layers):
            dropped_input = dropout(concatenated_inputs)
            gate_output = torch.sigmoid(gate(dropped_input))
            gated_input = concatenated_inputs * gate_output
            concat_input = torch.cat([concatenated_inputs, gated_input], dim=1)
            concatenated_inputs += dense(concat_input)

        outputs = [layer(concatenated_inputs) for layer in self.output_layers]
        output = torch.mean(torch.cat(outputs, dim=1), dim=1)
        return output
