import time
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from model import Model, ChessTrainDataset, ChessValidationDataset
from utils import stockfish_treshold, format_time


class Config:
    EPOCHS = 1000
    BATCH_SIZE = 128
    SAVE_PATH = './models/mlp-stockfish-1000.pth'


class ChessTrainer:
    def __init__(self, model, train_data, val_data, config):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def run_training_epoch(self):
        self.model.train()
        total_loss = 0
        batch_count = 0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device).float(), targets.apply_(stockfish_treshold).to(self.device).long()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        return total_loss / batch_count

    def run_validation_epoch(self):
        self.model.eval()
        total_loss = 0
        batch_count = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device).float(), targets.apply_(stockfish_treshold).to(self.device).long()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                batch_count += 1
        return total_loss / batch_count

    def train(self):
        self.train_loader = DataLoader(self.train_data, batch_size=self.config.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=self.config.BATCH_SIZE, shuffle=True)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.CrossEntropyLoss()

        for epoch in range(self.config.EPOCHS):
            start_time = time.time()
            train_loss = self.run_training_epoch()
            val_loss = self.run_validation_epoch()
            self.save_model(epoch)
            duration = time.time() - start_time
            remaining_time = (self.config.EPOCHS - epoch - 1) * duration

            print(f"Epoch {epoch+1}/{self.config.EPOCHS}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            print(f"Epoch Duration: {duration:.2f}s, Estimated Remaining Time: {format_time(remaining_time)}")

    def save_model(self, epoch):
        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer_dict': self.optimizer.state_dict()
        }, self.config.SAVE_PATH)

def evaluate_model(test_data):
    model = Model()
    checkpoint = torch.load('./models/mlp-stockfish-new.pth', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.3f}")

if __name__ == '__main__':
    model = Model()
    train_dataset = ChessTrainDataset()
    val_dataset = ChessValidationDataset()
    config = Config()

    trainer = ChessTrainer(model, train_dataset, val_dataset, config)
    trainer.train()
