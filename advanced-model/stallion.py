import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length].unsqueeze(-1)
        y = self.data[idx + self.seq_length]
        return x, y


class TimeSeriesModel(pl.LightningModule):
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super(TimeSeriesModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        pred = self.linear(lstm_out[:, -1, :])
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


if __name__ == '__main__':
    data = torch.sin(torch.linspace(0, 100, steps=1000))
    dataset = TimeSeriesDataset(data, seq_length=10)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = TimeSeriesModel(input_size=1)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_loader)
