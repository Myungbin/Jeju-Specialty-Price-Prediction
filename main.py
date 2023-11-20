import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader

from src.data.dataset import TabularDataset, TestDataset
from src.data.dataset import to_numpy
from src.data.preprocessing import DataPreprocessing, post_preprocessing
from src.inference.inference import inference
from src.model.gate_unit import TabularModel
from src.train.train import Trainer
from src.config.config import seed_everything, CFG

seed_everything(CFG.SEED)

train = pd.read_csv(r"./data/train.csv")
test = pd.read_csv(r'./data/test.csv')

preprocessing = DataPreprocessing(train, test)
x, y, test = preprocessing.fit()
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1103)
x_train, x_val, y_train, y_val = to_numpy(x, x_val, y, y_val)
train_dataset = TabularDataset(x_train, y_train)
val_dataset = TabularDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=CFG.BATCH_SIZE)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=CFG.BATCH_SIZE)

unique_value_list = x.nunique().tolist()
input_dims = [item + 1 for item in unique_value_list]



model = TabularModel(input_dims, 128)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-6,
                                                           last_epoch=-1)
trainer = Trainer(model, criterion, optimizer, scheduler)
trainer.fit(train_loader, val_loader)

# test_dataset = TestDataset(test)
# test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)

# model.load_state_dict(torch.load(r'./model.pth'))
# model.eval()
# pred_list = inference(model, test_loader)
# submission = pd.read_csv(r'./data/sample_submission.csv')
# submission['answer'] = pred_list
# submission = post_preprocessing(test, submission)
# submission.to_csv('./dnn_submission.csv', index=False)
