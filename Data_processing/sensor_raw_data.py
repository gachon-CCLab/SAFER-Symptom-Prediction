import pandas as pd
import numpy as np
from pandas import json_normalize
import json

class SensorDataProcessor:
    
    @staticmethod
    def load_sensing_data(file_path, chunksize=10000):
        """
        DataFrame에서 센서 데이터를 로드하고 처리하는 함수.
        
        매개변수:
        file_path (str): 센서 데이터 파일의 경로.
        chunksize (int): 데이터를 청크 단위로 나눌 크기.
        
        반환값:
        pd.DataFrame: 처리된 DataFrame.
        """
        # 빈 리스트를 생성하여 청크 단위로 처리된 데이터 프레임을 저장합니다.
        chunks = []
        
        # 청크 단위로 데이터를 읽고 처리합니다.
        for chunk in pd.read_csv(file_path, sep='\t', encoding='utf-8', chunksize=chunksize):
            # 필요한 컬럼들 정의
            chunk.columns = ['targetId', '이름', 'deviceId', '데이터', 'targetTime']
            
            # 불필요한 문자 제거 및 JSON 문자열 처리
            chunk['데이터'] = chunk['데이터'].str.replace('""', '"').str.replace('",', ',').str.strip(',')
            chunk = chunk.applymap(lambda x: x.strip(',') if isinstance(x, str) else x)

            # '데이터' 열의 JSON 데이터를 정규화
            def normalize_json(x):
                try:
                    return json.loads(x) if not pd.isna(x) else {}
                except json.JSONDecodeError:
                    return {}
            
            df_normalized = json_normalize(chunk['데이터'].apply(normalize_json))
            chunk = pd.concat([chunk.reset_index(drop=True), df_normalized], axis=1).drop(columns=['데이터'])
            
            # 'targetTime'을 datetime 형식으로 변환
            chunk['targetTime'] = pd.to_datetime(chunk['targetTime'], errors='coerce')  # 오류 발생 시 NaT로 변환
            
            chunks.append(chunk)
        
        # 모든 청크를 하나의 DataFrame으로 결합합니다.
        df = pd.concat(chunks, ignore_index=True)
        
        return df
    
    @staticmethod
    def process_sensing_data(combined_df):
        """
        센서 데이터를 처리하여 총 가속도(Total Acceleration)를 계산하고 불필요한 열을 제거하는 함수.
        
        매개변수:
        combined_df (pd.DataFrame): 입력 DataFrame.
        
        반환값:
        pd.DataFrame: 총 가속도 계산 및 불필요한 열이 제거된 DataFrame.
        """
        # 필요한 열이 모두 존재하는지 확인
        # 예시: 데이터프레임에 필요한 열이 있는지 확인
        required_columns = ['ACCELER_X_AXIS', 'ACCELER_Y_AXIS', 'ACCELER_Z_AXIS']
        missing_columns = [col for col in required_columns if col not in combined_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        
        if all(column in combined_df.columns for column in required_columns):
            # 총 가속도 계산
            combined_df['TOTAL_ACCELERATION'] = np.sqrt(
                combined_df['ACCELER_X_AXIS']**2 + 
                combined_df['ACCELER_Y_AXIS']**2 + 
                combined_df['ACCELER_Z_AXIS']**2
            )
        
        # 삭제할 열 목록 정의
        columns_to_drop = ['deviceId', 'ANGULAR_Y_AXIS', 'ANGULAR_X_AXIS', 'ANGULAR_Z_AXIS']
        
        # DataFrame에 존재하는 열만 삭제
        existing_columns_to_drop = [col for col in columns_to_drop if col in combined_df.columns]
        combined_df = combined_df.drop(columns=existing_columns_to_drop)
        
        return combined_df



    @staticmethod
    def aggregate_sensing_data(data):
        """
        센서 데이터를 1시간 간격으로 집계하는 함수.
        
        매개변수:
        data (pd.DataFrame): 처리된 센서 데이터를 포함한 DataFrame.
        
        반환값:
        pd.DataFrame: 계산된 통계를 포함한 집계된 DataFrame.
        """
        # Delta(처음과 마지막 값의 차이)를 계산하는 함수
        def delta(series):
            return series.iloc[-1] - series.iloc[0] if not series.empty else None
        
        agg_dict = {
            'TOTAL_ACCELERATION': ['first', 'last', 'mean', 'median', 'max', 'min', 'std', pd.Series.nunique],
            'HEARTBEAT': ['first', 'last', 'mean', 'median', 'max', 'min', 'std', pd.Series.nunique],
            'DISTANCE': delta,
            'SLEEP': delta,
            'STEP': delta,
            'CALORIES': delta
        }
        
        # 'targetTime'을 datetime 형식으로 변환하고 인덱스로 설정
        data['targetTime'] = pd.to_datetime(data['targetTime'])
        data.set_index('targetTime', inplace=True)
        
        # 데이터를 1분 간격으로 재샘플링하고 집계 사전을 적용
        result = data.groupby('이름').resample('1H').agg(agg_dict).reset_index()
        
        return result
    
    @staticmethod
    def reorganize_column_names(df):
        """
        다중 레벨의 열 이름을 평탄화하고, 각 열 이름의 순서를 재정렬하는 함수.
        
        매개변수:
        df (pd.DataFrame): 입력 DataFrame.
        
        반환값:
        pd.DataFrame: 재정렬된 열 이름을 가진 DataFrame.
        """
        # 다중 레벨의 열 이름을 평탄화
        df.columns = ['_'.join(col).strip() for col in df.columns.ravel()]

        # 각 열 이름 정리
        new_column_names = []
        for col in df.columns:
            parts = col.split('_')
            if len(parts) == 2:
                new_column_names.append(parts[1] + '_' + parts[0])
            elif len(parts) == 3:
                new_column_names.append(parts[2] + '_' + parts[0] + '_' + parts[1])
            else:
                new_column_names.append(col)
        
        df.columns = new_column_names
        
        # '이름'과 'targetTime' 열 이름 정리
        df = df.rename(columns={'_이름': '이름', '_targetTime': 'targetTime'})
        
        return df

"""
# 예시 사용:
# result = reorganize_column_names(result)
# print(result)

# 사용 예시:

# DataFrame `df`에 데이터를 가지고 있다고 가정:
df = pd.read_csv('/mount/nas/disk02/Data/Health/Mental_Health/SAFER/20240812/trait_state/seoul_first_half.csv', encoding='utf-8')
print(df)
# 센서 데이터 로드 및 처리
sensing_data = SensorDataProcessor.load_sensing_data(df)
print(sensing_data)
sensing_data = SensorDataProcessor.process_sensing_data(sensing_data)

# 처리된 센서 데이터를 집계
aggregated_data = SensorDataProcessor.aggregate_sensing_data(sensing_data)
aggregated_data = SensorDataProcessor.reorganize_column_names(aggregated_data)

# 최종 센서 데이터를 CSV로 저장 (선택 사항)
aggregated_data.to_csv('processed_sensing_data.csv', index=False)

# 처리되고 집계된 센서 데이터 출력
print(aggregated_data)
"""