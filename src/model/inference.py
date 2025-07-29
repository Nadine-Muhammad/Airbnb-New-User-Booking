import torch
import torch.nn as nn
import random
import pandas as pd
import os
import pickle
from tabulate import tabulate
from src.preprocess import ProcessingPipeline
from src.model.model import AirbnbNN 

base_dir = os.getcwd()
processed_dir = os.path.join(base_dir, 'data', 'processed')
test_set = pd.read_csv(os.path.join(processed_dir, 'test.csv'))

model_folder = os.path.join(base_dir, 'src', 'model')
model = AirbnbNN()
model_path = os.path.join(model_folder, 'model-0.1.0.pth')
model.load_state_dict(torch.load(model_path))
model.eval()

mappings_path = os.path.join(model_folder, 'mappings.pkl')
with open(mappings_path, "rb") as pickle_file:
        mappings = pickle.load(pickle_file)

def predict_random_record(dataset= test_set, mappings= mappings, model=model):
    """
    Gets a random record from test dataset and applies preprocessing pipeline.
    Makes a prediction and returns the predicted label name and randomly selected row printed by tabulate.
    Args:
        dataset (pd.DataFrame): defaults to test users dataset.
        mappings (Dict): defaults to loaded class mappings.
        model: defaults to loaded model.
    Returns:
        tuple: Random record, model prediction.
    """
    random_index = random.randint(0, len(dataset) - 1)
    random_record = dataset.iloc[[random_index]]

    pipeline = ProcessingPipeline(random_record)
    input_tensor = pipeline.get_tensor() 
    
    with torch.no_grad():
        prediction = model(input_tensor) #Returns probabilities for each class

    predicted_classes = torch.argmax(prediction, dim=1) #Returns the encoded label of predicted class
    label = mappings[predicted_classes.item()] #Maps encoded label to actual class name

    headers = list(dataset.columns)
    row = tabulate(dataset.iloc[[random_index]], headers=headers, tablefmt="pretty")

    return row, label