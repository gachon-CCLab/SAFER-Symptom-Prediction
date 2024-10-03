import pandas as pd

class DataProcessor:
    def __init__(self, location_data=None, sensor_data=None, crf_data=None, trait_data=None):
        """
        DataProcessor 클래스 생성자.
        데이터는 DataFrame으로 직접 전달하거나 나중에 load_data 메소드를 통해 로드할 수 있습니다.
        """
        self.location_data = location_data
        self.sensor_data = sensor_data
        self.crf_data = crf_data
        self.trait_data = trait_data
        self.merged_data = None

    def load_data(self, location_file=None, sensor_file=None, crf_file=None, trait_file=None):
        """
        CSV 파일을 로드하여 DataFrame으로 저장하는 함수.
        """
        if isinstance(location_file, str) and location_file:
            self.location_data = pd.read_csv(location_file)
        elif isinstance(location_file, pd.DataFrame):
            self.location_data = location_file
        
        if isinstance(sensor_file, str) and sensor_file:
            self.sensor_data = pd.read_csv(sensor_file)
        elif isinstance(sensor_file, pd.DataFrame):
            self.sensor_data = sensor_file

        if isinstance(crf_file, str) and crf_file:
            self.crf_data = pd.read_csv(crf_file, encoding='utf-8')
        elif isinstance(crf_file, pd.DataFrame):
            self.crf_data = crf_file

        if isinstance(trait_file, str) and trait_file:
            self.trait_data = pd.read_csv(trait_file, encoding='utf-8')
        elif isinstance(trait_file, pd.DataFrame):
            self.trait_data = trait_file

    def merge_location_and_sensor(self):
        """
        위치 데이터와 센서 데이터를 공통 컬럼으로 병합하는 함수.
        """
        if self.location_data is None or self.sensor_data is None:
            raise ValueError("Location data or Sensor data is missing.")
        
        # Ensure both dataframes have the same 'targetTime' dtype
        self.location_data['targetTime'] = pd.to_datetime(self.location_data['targetTime'])
        self.sensor_data['targetTime'] = pd.to_datetime(self.sensor_data['targetTime'])
        
        # Merge the location and sensor dataframes on '이름' and 'targetTime'
        self.merged_data = pd.merge(self.location_data, self.sensor_data, on=['이름', 'targetTime'], how='inner')
        
        # Debugging: Output the merged data length
        print(f"Merged data length: {len(self.merged_data)}")

        # Optionally, inspect the data
        print(self.merged_data.head())

        # If there are mismatches, you can check where they occur
        if len(self.merged_data) != len(self.location_data):
            print("Mismatch in merging: ")
            print("Location data head:\n", self.location_data.head())
            print("Sensor data head:\n", self.sensor_data.head())

    def process_crf_data(self):
        """
        CRF 데이터를 처리하고 merged_data와 병합하는 함수.
        """
        if self.crf_data is None or self.merged_data is None:
            raise ValueError("CRF 데이터 또는 병합된 데이터가 없습니다.")

        # 불필요한 열 제거 및 열 이름 변경
        if 'Unnamed: 0' in self.crf_data.columns:
            self.crf_data = self.crf_data.drop(columns=['Unnamed: 0'])
        if 'researchNum' in self.crf_data.columns:
            self.crf_data = self.crf_data.drop(columns=['researchNum'])
        
        self.crf_data = self.crf_data.rename(columns={'key': '이름'})
        
        drop_columns = ['VE_tx', 'VE_tx_inj_time', 'VE_tx_secl_startime', 'VE_tx_secl_endtime', 
                        'VE_tx_rest_startime', 'VE_tx_rest_endtime', 'week', 'Eval_datetime', 'elapsed_date']
        existing_columns = [col for col in drop_columns if col in self.crf_data.columns]
        if existing_columns:
            self.crf_data = self.crf_data.drop(columns=existing_columns)
        
        # 'VE_tx_time'을 datetime으로 변환
        self.crf_data['VE_tx_time'] = pd.to_datetime(self.crf_data['VE_tx_time'])
        
        # 병합된 데이터 정렬
        self.merged_data.sort_values(by=['이름', 'targetTime'], inplace=True)
        
        # CRF 데이터와 merged_data 병합
        for key, group in self.merged_data.groupby('이름'):
            first_date = group['targetTime'].min()
            crf_subset = self.crf_data[self.crf_data['이름'] == key].reset_index(drop=True)
            
            #7일 간격으로 mapping
            for i, crf_row in crf_subset.iterrows():
                start_date = first_date + pd.Timedelta(days=7 * i)
                end_date = start_date + pd.Timedelta(days=7)
                mask = (self.merged_data['이름'] == key) & (self.merged_data['targetTime'] >= start_date) & (self.merged_data['targetTime'] < end_date)
                for col in crf_row.index:
                    if col not in ['이름', 'targetTime']:
                        self.merged_data.loc[mask, col] = crf_row[col]
            
            # 응급 상황 처리-> 기록지 생성 일자에 맞춰 매핑
            emergency_data = crf_subset[crf_subset['status'] == 'emergency']
            for emg_row in emergency_data.itertuples():
                date_match = (self.merged_data['이름'] == key) & (self.merged_data['targetTime'].dt.date == getattr(emg_row, 'VE_tx_time').date())
                if date_match.any():
                    for col in crf_subset.columns.difference(['이름', 'VE_tx_time', 'status']):
                        self.merged_data.loc[date_match, col] = getattr(emg_row, col)
        
        # 'idxInfo' 열이 존재하면 제거
        if 'idxInfo' in self.merged_data.columns:
            self.merged_data = self.merged_data.drop(columns=['idxInfo'])

    def merge_trait_data(self):
        """
        성향 데이터를 병합된 데이터에 병합하는 함수.
        """
        if self.trait_data is None or self.merged_data is None:
            raise ValueError("성향 데이터 또는 병합된 데이터가 없습니다.")
        
        self.trait_data = self.trait_data.rename(columns={'key': '이름'})
        trait_subset = self.trait_data
        
        self.merged_data = pd.merge(self.merged_data, trait_subset, on='이름')
        
        return self.merged_data
