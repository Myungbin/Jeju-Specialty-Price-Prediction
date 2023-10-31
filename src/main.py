import pandas as pd
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from src.data.dataset import TabularDataset, TestDataset
from src.data.preprocessing import preprocessing
from src.model.gate_unit import TabularModel
from src.train.train import Trainer
from src.inference.inference import inference
import torch
train = pd.read_csv(r"C:\Project\jeju-price-prediction\data\train.csv")
test = pd.read_csv(r'C:\Project\jeju-price-prediction\data\test.csv')
x, y, test = preprocessing(train, test)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1103)
x_train = x_train.values
x_val = x_val.values
y_train = y_train.values
y_val = y_val.values

train_dataset = TabularDataset(x_train, y_train)
val_dataset = TabularDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=64)

input_dims = [6, 7, 3, 6, 13, 32, 8, 3, 40, 25, 5, 3]
model = TabularModel(input_dims, 12)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=3e-4)
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
# trainer = Trainer(model, criterion, optimizer, scheduler)
# trainer.fit(train_loader, val_loader)
#
test_dataset = TestDataset(test)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)

model.load_state_dict(torch.load(r'C:\Project\jeju-price-prediction\src\model\best_model.pth'))
model.eval()
pred_list = inference(model, test_loader)
submission = pd.read_csv(r'C:\Project\jeju-price-prediction\data\sample_submission.csv')
submission['answer'] = pred_list
submission.to_csv('./dnn_submission.csv', index=False)