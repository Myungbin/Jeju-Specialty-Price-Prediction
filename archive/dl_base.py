`import holidays
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
        # embeddings = []
        # for i, e in enumerate(self.embeddings):
        #     embeddings.append(e(x[:, i]))
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
    def __init__(self, model, criterion, optimizer, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_loss = 99999

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

            if self.scheduler is not None:
                self.scheduler.step()

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')


train = pd.read_csv(r'./data/train.csv')
test = pd.read_csv(r'./data/test.csv')
international_trade = pd.read_csv(r'./data/international_trade.csv')


def group_season(df):
    df.loc[(df['month'] == 3) | (df['month'] == 4) | (df['month'] == 5), 'season'] = '봄'
    df.loc[(df['month'] == 6) | (df['month'] == 7) | (df['month'] == 8), 'season'] = '여름'
    df.loc[(df['month'] == 9) | (df['month'] == 10) | (df['month'] == 11), 'season'] = '가을'
    df.loc[(df['month'] == 12) | (df['month'] == 1) | (df['month'] == 2), 'season'] = '겨울'
    return df['season']


def holiday(df):
    kr_holidays = holidays.KR()
    df['holiday'] = df.timestamp.apply(lambda x: 'holiday' if x in kr_holidays else 'non-holiday')
    return df['holiday']


train['year'] = train['timestamp'].apply(lambda x: int(x[0:4]))
train['month'] = train['timestamp'].apply(lambda x: int(x[5:7]))
train['day'] = train['timestamp'].apply(lambda x: int(x[8:10]))
train['Weekday'] = pd.to_datetime(train['timestamp']).dt.weekday
train['is_weekend'] = train['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
train['year'] = train['year'] - 2019

test['year'] = test['timestamp'].apply(lambda x: int(x[0:4]))
test['month'] = test['timestamp'].apply(lambda x: int(x[5:7]))
test['day'] = test['timestamp'].apply(lambda x: int(x[8:10]))
test['Weekday'] = pd.to_datetime(test['timestamp']).dt.weekday
test['is_weekend'] = test['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
test['year'] = test['year'] - 2019

train['season'] = group_season(train)
test['season'] = group_season(test)

train['holiday'] = holiday(train)
test['holiday'] = holiday(test)

x = train.drop(columns=['ID', 'timestamp', 'supply(kg)', 'price(원/kg)'])
y = train['price(원/kg)']

x_test = test.drop(columns=['ID', 'timestamp'])

qual_col = ['item', 'corporation', 'location', 'season', 'holiday']

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

train_loader = DataLoader(train_dataset, shuffle=False, batch_size=64)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=64)

input_dims = [6, 7, 3, 6, 13, 32, 8, 3, 5, 3]
model = TabularModel(input_dims, 10)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1, verbose=True)

trainer = Trainer(model, criterion, optimizer, scheduler)

trainer.fit(train_loader, val_loader)
