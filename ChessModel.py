import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class ChessValidationDataset(Dataset):
    def __init__(self, file_path='./data/stockfish_processed1M.npz', val_split=150_000):
        data = np.load(file_path)
        self.input_data = data['arr_0'][:val_split]
        self.output_data = data['arr_1'][:val_split]
        print(f'Validation Data Loaded: Inputs shape {self.input_data.shape}, Outputs shape {self.output_data.shape}')

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        return self.input_data[index], self.output_data[index]

class ChessTrainDataset(Dataset):
    def __init__(self, file_path='./data/stockfish_processed1M.npz', val_split=150_000):
        data = np.load(file_path)
        self.input_data = data['arr_0'][val_split:]
        self.output_data = data['arr_1'][val_split:]
        print(f'Training Data Loaded: Inputs shape {self.input_data.shape}, Outputs shape {self.output_data.shape}')

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        return self.input_data[index], self.output_data[index]

class ChessNet(nn.Module):
    def __init__(self, input_size=768, hidden_size1=256, hidden_size2=64, num_classes=3, dropout_prob=0.5):
        super(ChessNet, self).__init__()
        self.fc_layer1 = nn.Linear(input_size, hidden_size1)
        self.fc_layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc_layer3 = nn.Linear(hidden_size2, num_classes)
        self.dropout_layer = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.leaky_relu(self.fc_layer1(x))
        x = self.dropout_layer(x)
        x = F.leaky_relu(self.fc_layer2(x))
        x = self.dropout_layer(x)
        x = self.fc_layer3(x)
        return x

if __name__ == '__main__':
    train_data = ChessTrainDataset()
    val_data = ChessValidationDataset()
    chess_model = ChessNet()

    # Example usage
    print(f'Model Structure: {chess_model}')
    print(f'Number of Training Samples: {len(train_data)}')
    print(f'Number of Validation Samples: {len(val_data)}')
