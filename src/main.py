import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader

from src.data.dataset import TabularDataset
from src.data.dataset import to_numpy
from src.data.preprocessing import DataPreprocessing
from src.model.gate_unit import TabularModel
from src.train.train import Trainer

train = pd.read_csv(r"C:\Project\jeju-price-prediction\data\train.csv")
test = pd.read_csv(r'C:\Project\jeju-price-prediction\data\test.csv')

preprocessing = DataPreprocessing(train, test)
x, y, test = preprocessing.fit()
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1103)
x_train, x_val, y_train, y_val = to_numpy(x_train, x_val, y_train, y_val)
train_dataset = TabularDataset(x_train, y_train)
val_dataset = TabularDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=128)

unique_value_list = x.nunique().tolist()
input_dims = [item + 1 for item in unique_value_list]

model = TabularModel(input_dims, 256)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-6, last_epoch=-1)
trainer = Trainer(model, criterion, optimizer, scheduler)
trainer.fit(train_loader, val_loader)

# test_dataset = TestDataset(test)
# test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)
#
# model.load_state_dict(torch.load(r'C:\Project\jeju-price-prediction\src\model.pth'))
# model.eval()
# pred_list = inference(model, test_loader)
# submission = pd.read_csv(r'C:\Project\jeju-price-prediction\data\sample_submission.csv')
# submission['answer'] = pred_list
# submission.to_csv('./dnn_submission.csv', index=False)
