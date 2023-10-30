import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TabularDataset(Dataset):
    def __init__(self, x, y) -> None:
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y


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
            concatenated_inputs = concatenated_inputs + dense(concat_input)

        outputs = [layer(concatenated_inputs) for layer in self.output_layers]
        output = torch.mean(torch.cat(outputs, dim=1), dim=1)
        return output


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred, y):
        loss = torch.sqrt(self.mse(y_pred, y) + self.eps)
        return loss


class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train_step(self, train_loader):
        self.model.train()
        train_loss = 0
        for data, label in tqdm(train_loader, leave=False):
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output.to(torch.float32), label.to(torch.float32))

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        avg_train_loss = (train_loss / len(train_loader)) ** 0.5

        return avg_train_loss

    def validation_step(self, validation_loader):
        val_loss = 0
        with torch.no_grad():
            for data, label in tqdm(validation_loader, leave=False):
                prediction = self.model(data)
                loss = self.criterion(prediction.to(torch.float32), label.to(torch.float32))
                val_loss += loss.item()
            avg_validation_loss = (val_loss / len(validation_loader)) ** 0.5

        return avg_validation_loss

    def fit(self, train_loader, val_loader):
        for epoch in range(100):
            train_loss = self.train_step(train_loader)
            val_loss = self.validation_step(val_loader)

            print(f"Epoch [{epoch + 1}/{100}]"
                  f"Training Loss: {train_loss:.7f} "
                  f"Validation Loss: {val_loss:.7f} ")


train = pd.read_csv(r'./data/train.csv')
test = pd.read_csv(r'./data/test.csv')
international_trade = pd.read_csv(r'./data/international_trade.csv')

train['year'] = train['timestamp'].apply(lambda x: int(x[0:4]))
train['month'] = train['timestamp'].apply(lambda x: int(x[5:7]))
train['day'] = train['timestamp'].apply(lambda x: int(x[8:10]))

test['year'] = test['timestamp'].apply(lambda x: int(x[0:4]))
test['month'] = test['timestamp'].apply(lambda x: int(x[5:7]))
test['day'] = test['timestamp'].apply(lambda x: int(x[8:10]))

x = train.drop(columns=['ID', 'timestamp', 'supply(kg)', 'price(원/kg)'])
y = train['price(원/kg)']

x_test = test.drop(columns=['ID', 'timestamp'])

qual_col = ['item', 'corporation', 'location']

for i in qual_col:
    le = LabelEncoder()
    x[i] = le.fit_transform(x[i])
    x_test[i] = le.transform(x_test[i])

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1103)

x_train = x_train.values
x_val = x_val.values
y_train = y_train.values
y_val = y_val.values

train_dataset = TabularDataset(x_train, y_train)
val_dataset = TabularDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=512)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=512)

input_dims = [6, 7, 3, 2024, 13, 31]
model = TabularModel(input_dims, 6)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
trainer = Trainer(model, criterion, optimizer)
trainer.fit(train_loader, val_loader)
