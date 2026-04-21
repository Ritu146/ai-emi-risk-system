import torch
import joblib
import numpy as np
from model import Model

# Load once
scaler = joblib.load("scaler.pkl")

model = Model(11)  # change to your actual input features
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

def predict(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    return model(input_tensor).item()
