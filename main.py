
from time import time
import joblib
import torch
import webdataset as wds
from sklearn.preprocessing import StandardScaler
import numpy as np
import glob
import multiprocessing as mp
import time
from torch.utils.data import Dataset, DataLoader
import pickle


class TorchScaler:
    def __init__(self, path=None):
        self.mean = None
        self.var = None
        if path:
            self.scaler = self._load(path)
        else:
            self.scaler = StandardScaler()
    
    def parital_fit(self, data):
        data = self._reshape_data(data)
        self.scaler.partial_fit(data)
    
    def transform(self, data):
        data = self._reshape_data(data)
        data -= torch.tensor(self.scaler.mean_).to("cuda")
        data /= torch.tensor(self.scaler.var_).to("cuda")
        return data

    def _reshape_data(self, data):
        return data.reshape(-1, data.shape[-1])

    def fit_from_loader(self, dataloader: DataLoader):
        for data in dataloader:
            self.parital_fit(data)
    
    def save(self, path):
        joblib.dump(self.scaler, path) 

    def _load(self, path):
        return joblib.load(path)


class TestDataset(Dataset):
    def __init__(self):
        self.x = torch.randn((100000, 128, 5))

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.x.shape[0]


if __name__ == '__main__':   
    dataset = TestDataset()
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Fit scaler
    ts = TorchScaler()
    ts.fit_from_loader(train_loader)
    ts.save("my_scaler.pickle")

    # Load scaler
    ts = TorchScaler("my_scaler.pickle")
    data = torch.randn(1, 128, 5)
    data = ts.transform(data)