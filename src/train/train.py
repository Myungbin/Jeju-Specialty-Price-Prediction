from tqdm import tqdm
import torch


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
