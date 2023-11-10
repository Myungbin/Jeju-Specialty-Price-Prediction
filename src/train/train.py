import torch
from tqdm import tqdm

from src.config.config import CFG


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler=None):
        self.model = model.to(CFG.DEVICE)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_loss = 100000

    def train_step(self, train_loader):
        self.model.train()
        train_loss = 0
        for data, label in tqdm(train_loader, leave=False):
            data, label = data.to(CFG.DEVICE), label.to(CFG.DEVICE)
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
                data, label = data.to(CFG.DEVICE), label.to(CFG.DEVICE)
                prediction = self.model(data)
                loss = self.criterion(prediction.to(torch.float32), label.to(torch.float32))
                val_loss += loss.item()
            avg_validation_loss = (val_loss / len(validation_loader)) ** 0.5

        return avg_validation_loss

    def fit(self, train_loader, val_loader):
        for epoch in range(CFG.EPOCHS):
            train_loss = self.train_step(train_loader)
            val_loss = self.validation_step(val_loader)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), './model.pth')

            print(f"Epoch [{epoch + 1}/{100}]"
                  f"Training Loss: {train_loss:.7f} "
                  f"Validation Loss: {val_loss:.7f} ")
