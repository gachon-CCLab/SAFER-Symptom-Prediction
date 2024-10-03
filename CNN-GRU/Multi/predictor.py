import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle  # pickle import 추가
from model import CNNGRUModel
from data_loader import  DataProcessor  # DataProcessor는 그대로 사용
import pkg_resources
class Predictor:
    def __init__(self, device, model_path=None):
        """
        모델과 장치를 초기화합니다.
        """
        
        if model_path is None :
            model_path = pkg_resources.resource_filename('gsecure', './model2/model/final_model.pkl')
        # 기본 seq_cols 및 target_cols 고정
        self.seq_cols = [
            'Daily_Entropy', 'Normalized_Daily_Entropy', 'Eight_Hour_Entropy', 'Normalized_Eight_Hour_Entropy',
            'first_TOTAL_ACCELERATION', 'Location_Variability', 'last_TOTAL_ACCELERATION', 'mean_TOTAL_ACCELERATION',
            'median_TOTAL_ACCELERATION', 'max_TOTAL_ACCELERATION', 'min_TOTAL_ACCELERATION', 'std_TOTAL_ACCELERATION',
            'nunique_TOTAL_ACCELERATION', 'delta_CALORIES', 'first_HEARTBEAT', 'last_HEARTBEAT', 'mean_HEARTBEAT',
            'median_HEARTBEAT', 'max_HEARTBEAT', 'min_HEARTBEAT', 'std_HEARTBEAT', 'nunique_HEARTBEAT',
            'delta_DISTANCE', 'delta_SLEEP', 'delta_STEP', 'sex', 'age', 'place_Unknown', 'place_hallway', 'place_other', 'place_ward'
        ]
        self.target_cols = ['BPRS_sum']
        # self.target_cols = ['YMRS_sum']
        # self.target_cols = ['MADRS_sum']
        # self.target_cols = ['HAMA_sum']
       

        self.device = device
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()  # 평가 모드로 전환

    def load_model(self, model_path):
        """
        저장된 pickle 모델을 로드합니다.
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def preprocess_data(self, data_path):
        """
        데이터를 로드하고 전처리한 후 DataLoader로 반환합니다.
        """
        data = pd.read_csv(data_path)
        data = DataProcessor.preprocess_data(data)
        data = DataProcessor.reset_week_numbers(data)
        data = DataProcessor.transform_target(data)

        max_length = DataProcessor.find_max_sequence_length_by_week(data, self.seq_cols)

        # 데이터를 모델에 맞게 준비하고 텐서로 변환
        results = DataProcessor.prepare_data_for_model_by_week(data, max_length, self.seq_cols, self.target_cols)
        X_tensor, _ = DataProcessor.convert_results_to_tensors(results)
        return DataLoader(TensorDataset(X_tensor), batch_size=16, shuffle=False)

    def predict(self, data_loader):
        """
        데이터를 예측하고 결과를 반환합니다.
        """
        predictions = []
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs[0].to(self.device)  # 입력 텐서로 변환
                outputs = self.model(inputs)
                # 확률값을 이진값(0 또는 1)으로 변환
                binary_predictions = (outputs >= 0.5).cpu().numpy()
                predictions.extend(binary_predictions)

        return np.array(predictions)


    # 디버깅할 때 사용하는 함수라 필요 없음
    def save_predictions(self, predictions, output_path):
        """
        예측 결과를 파일로 저장합니다.
        """
        df = pd.DataFrame(predictions, columns=self.target_cols)
        df.to_csv(output_path, index=False)
        print(f'Predictions saved at {output_path}')
