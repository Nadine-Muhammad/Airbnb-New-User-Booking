import pandas as pd
import torch
import os
import pickle

base_dir = os.getcwd()
model_path = os.path.join(base_dir, 'src', 'model')
encoder_path = os.path.join(model_path, 'target_encoder.pkl')
scaler_path = os.path.join(model_path, 'scaler.pkl')

with open(encoder_path, "rb") as file:
    encodings = pickle.load(file)

with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

"""
Preprocessing pipeline to prepare test data for inference.
Target-encodes categorical features, scales them using a saved MinMaxScaler, and transforms to tensors.
"""

class ProcessingPipeline():
    def __init__(self, data: pd.DataFrame) -> None: 
            self.data = data
            self.target_encode()
            self.data = scaler.transform(self.data)         


    def target_encode(self, encodings=encodings) -> pd.DataFrame:
        for column, encoding_dict in encodings.items():
            if column in self.data.columns:
                self.data[column] = self.data[column].replace(encoding_dict)


    def get_tensor(self) -> torch.Tensor:
        return torch.tensor(self.data, dtype=torch.float32)            