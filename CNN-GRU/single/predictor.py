import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle  # Added pickle import
from model import CNNGRUModel
from data_loader import DataProcessor  # DataProcessor is used as is
import pkg_resources

class Predictor:
    def __init__(self, device, model_path=None):
        """
        Initializes the model and device.
        """
        
        if model_path is None:
            model_path = pkg_resources.resource_filename('gsecure', './model/model.pkl')
        # Default seq_cols and target_cols are fixed
        self.seq_cols = [
            'Daily_Entropy', 'Normalized_Daily_Entropy', 'Eight_Hour_Entropy', 'Normalized_Eight_Hour_Entropy',
            'first_TOTAL_ACCELERATION', 'Location_Variability', 'last_TOTAL_ACCELERATION', 'mean_TOTAL_ACCELERATION',
            'median_TOTAL_ACCELERATION', 'max_TOTAL_ACCELERATION', 'min_TOTAL_ACCELERATION', 'std_TOTAL_ACCELERATION',
            'nunique_TOTAL_ACCELERATION', 'delta_CALORIES', 'first_HEARTBEAT', 'last_HEARTBEAT', 'mean_HEARTBEAT',
            'median_HEARTBEAT', 'max_HEARTBEAT', 'min_HEARTBEAT', 'std_HEARTBEAT', 'nunique_HEARTBEAT',
            'delta_DISTANCE', 'delta_SLEEP', 'delta_STEP', 'sex', 'age', 'place_Unknown', 'place_hallway', 'place_other', 'place_ward'
        ]
        self.target_cols = ['BPRS_sum', 'YMRS_sum', 'MADRS_sum', 'HAMA_sum']

        self.device = device
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()  # Switch to evaluation mode

    def load_model(self, model_path):
        """
        Loads the saved pickle model.
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def preprocess_data(self, data_path):
        """
        Loads and preprocesses data, returning it as a DataLoader.
        """
        data = pd.read_csv(data_path)
        data = DataProcessor.preprocess_data(data)
        data = DataProcessor.reset_week_numbers(data)
        data = DataProcessor.transform_target(data)

        max_length = DataProcessor.find_max_sequence_length_by_week(data, self.seq_cols)

        # Prepare the data for the model and convert to tensors
        results = DataProcessor.prepare_data_for_model_by_week(data, max_length, self.seq_cols, self.target_cols)
        X_tensor, _ = DataProcessor.convert_results_to_tensors(results)
        return DataLoader(TensorDataset(X_tensor), batch_size=16, shuffle=False)

    def predict(self, data_loader):
        """
        Predicts outcomes based on the input data and returns the results.
        """
        predictions = []
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs[0].to(self.device)  # Convert to input tensor
                outputs = self.model(inputs)
                # Convert probabilities to binary values (0 or 1)
                binary_predictions = (outputs >= 0.5).cpu().numpy()
                predictions.extend(binary_predictions)

        return np.array(predictions)


    # This function is used for debugging, not necessary
    def save_predictions(self, predictions, output_path):
        """
        Saves the prediction results to a file.
        """
        df = pd.DataFrame(predictions, columns=self.target_cols)
        df.to_csv(output_path, index=False)
        print(f'Predictions saved at {output_path}')
