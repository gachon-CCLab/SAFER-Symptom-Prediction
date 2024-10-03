import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class LocationProcessor:
    @staticmethod
    def load_data_from_csv(file_path):
        """
        CSV 파일을 로드하여 전처리하는 함수.
        :param file_path: CSV 파일 경로
        :return: 전처리된 DataFrame
        """
        try:
            data = pd.read_csv(file_path, sep='\t',encoding='utf-8')
            print(f"CSV 파일 {file_path}이 성공적으로 로드되었습니다.")
            
            # 위치 데이터 로드 및 전처리
            data = LocationProcessor.load_location_data(data)
            data = LocationProcessor.preprocess_location_data(data)
            data = LocationProcessor.resample_and_calculate(data)
            
            print("데이터 전처리가 완료되었습니다.")
            return data
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {file_path}")
        except pd.errors.EmptyDataError:
            print("파일이 비어 있습니다.")
        except pd.errors.ParserError:
            print("파일을 파싱하는 중 오류가 발생했습니다.")
        except Exception as e:
            print(f"파일 로드 중 오류 발생: {e}")
        return None
    
    @staticmethod
    def load_location_data(data):
        """
        위치 데이터를 DataFrame으로 로드하는 함수.
        :param data: 위치 데이터가 포함된 원본 DataFrame
        :return: 전처리된 DataFrame
        """
        try:
            # 불필요한 열 제거
            columns_to_drop = [col for col in data.columns if '\t' in col]
            data = data.drop(columns=columns_to_drop)
            
            # 열 이름 지정
            data.columns = ['No.', '디바이스 이름', '이름', '위도', '경도', 'targetTime']
            data = data.applymap(lambda x: x.strip(',') if isinstance(x,str) else x)
            # 불필요한 열 삭제
            data = data.drop(columns=['No.', '디바이스 이름'])
            print("위치 데이터가 성공적으로 로드되었습니다.")
            return data
        except KeyError as e:
            print(f"키 오류: {e}")
        except Exception as e:
            print(f"위치 데이터 로드 중 오류 발생: {e}")
        return data
    
    @staticmethod
    def preprocess_location_data(data):
        """
        위치 데이터를 전처리하는 함수.
        'targetTime'을 datetime으로 변환하고, 인덱스를 설정합니다.
        :param data: 위치 데이터 DataFrame
        :return: 전처리된 DataFrame
        """
        try:
            data['targetTime'] = pd.to_datetime(data['targetTime'])
            data.set_index(['이름', 'targetTime'], inplace=True)
            
            print("위치 데이터 전처리가 완료되었습니다.")
            return data
        except KeyError as e:
            print(f"키 오류: {e}")
        except Exception as e:
            print(f"위치 데이터 전처리 중 오류 발생: {e}")
        return data
    
    @staticmethod
    def calculate_mode(df):
        """
        주어진 DataFrame의 최빈값을 계산하는 함수.
        :param df: DataFrame
        :return: 최빈값 Series
        """
        modes = df.mode()
        if not modes.empty:
            return modes.iloc[0]
        else:
            return None
    
    @staticmethod
    def calculate_entropy(group):
        """
        그룹 내 엔트로피 및 정규화된 엔트로피를 계산하는 함수.
        :param group: DataFrame 그룹
        :return: 엔트로피와 정규화된 엔트로피 (각각 NaN일 수 있음)
        """
        if len(group) == 0:
            return np.nan, np.nan
        time_spent = group.groupby('이름').size() / group.shape[0]
        entropy = -sum(time_spent * np.log(time_spent + 1e-10))  # log(0)을 피하기 위해 1e-10 추가
        max_entropy = np.log(len(group['이름'].unique())) if group['이름'].nunique() > 1 else 1
        normalized_entropy = entropy / max_entropy
        return entropy, normalized_entropy
    
    @staticmethod
    def calculate_location_variance(group):
        """
        위치 데이터의 위도 및 경도에 대한 분산을 계산하는 함수.
        :param group: DataFrame 그룹
        :return: 위도 및 경도 분산의 로그 값
        """
        lat_var = np.var(group['위도'].astype(float))  # 문자열을 float으로 변환
        lon_var = np.var(group['경도'].astype(float))  # 문자열을 float으로 변환
        return np.log(lat_var + lon_var + 1e-10)  # log(0)을 피하기 위해 1e-10 추가
    
    @staticmethod
    def sliding_window_variability(df, window_size):
        variability = []
        for start in range(0, len(df)- window_size + 1, window_size):
            window = df.iloc[start:start+window_size]
            if len(window) > 1:
                var = LocationProcessor.calculate_location_variance(window)
                variability.append(var)
            else :
                variability.append(np.nan)

        return variability
    
    @staticmethod
    def resample_and_calculate(data):
        """
        데이터를 리샘플링하고, 엔트로피 및 위치 가변성을 계산하는 함수.
        :param data: 전처리된 위치 데이터 DataFrame
        :return: 리샘플링 및 계산된 엔트로피 및 위치 가변성을 포함한 DataFrame
        """
        try:
            df_minute_mode = data.groupby(level='이름').resample('1H', level='targetTime').apply(LocationProcessor.calculate_mode)
            df_minute_mode = df_minute_mode.reset_index()
            df_minute_mode['Date'] = df_minute_mode['targetTime'].dt.date
            
            # 24시간 엔트로피 계산
            daily_entropy_df = data.reset_index()
            daily_entropy_df['Date'] = daily_entropy_df['targetTime'].dt.date
            daily_entropy_result = daily_entropy_df.groupby('Date').apply(LocationProcessor.calculate_entropy).apply(pd.Series).reset_index()
            daily_entropy_result.columns = ['Date', 'Daily_Entropy', 'Normalized_Daily_Entropy']
            df_minute_mode = df_minute_mode.merge(daily_entropy_result, on='Date', how='left')
            
            # 8시간 엔트로피 계산
            eight_hour_entropy_df = data.reset_index()
            eight_hour_entropy_result = eight_hour_entropy_df.groupby(pd.Grouper(key='targetTime', freq='8H')).apply(LocationProcessor.calculate_entropy).apply(pd.Series).reset_index()
            eight_hour_entropy_result.columns = ['targetTime', 'Eight_Hour_Entropy', 'Normalized_Eight_Hour_Entropy']
            df_minute_mode = df_minute_mode.merge(eight_hour_entropy_result, on='targetTime', how='left')
            
            location_variability = LocationProcessor.sliding_window_variability(df_minute_mode,8)
            df_minute_mode['Location_Variability'] = pd.Series(location_variability,index = df_minute_mode.index[:len(location_variability)])
            print("리샘플링 및 엔트로피 계산이 완료되었습니다.")
            
            return df_minute_mode
        except Exception as e:
            print(f"리샘플링 및 계산 중 오류 발생: {e}")
        return data
    
    @staticmethod
    def assign_location_labels(data, location_dict):
        """
        위치 데이터를 기준으로 가장 가까운 위치 레이블을 할당하는 함수.
        :param data: 위치 데이터 DataFrame
        :param location_dict: 위치와 좌표를 매핑하는 딕셔너리
        :return: 위치 레이블이 할당된 DataFrame
        """
        try:
            data = data.fillna(0)
            # '위도'와 '경도'를 숫자형으로 변환 (에러 시 NaN으로 처리)
            data['위도'] = pd.to_numeric(data['위도'], errors='coerce')
            data['경도'] = pd.to_numeric(data['경도'], errors='coerce')

            # NaN 값을 0으로 대체
            data = data.fillna(0)

            # location_dict에서 좌표 데이터 추출
            coords = np.array(list(location_dict.keys()))

            # NearestNeighbors 모델 생성 및 학습
            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(coords)

            def get_nearest_location(row):
                # 각 row의 위도와 경도에 대한 가장 가까운 위치 계산
                distance, index = neigh.kneighbors([[row['위도'], row['경도']]])
                nearest_coord = coords[index[0][0]]
                return location_dict[tuple(nearest_coord)]

            # 'place' 열에 위치 이름 매핑
            data['place'] = data.apply(get_nearest_location, axis=1)

            # '위도'와 '경도' 열 삭제
            data = data.drop(columns=['위도', '경도'])

            print("위치 레이블 할당이 완료되었습니다.")
            return data
        except KeyError as e:
            print(f"키 오류: {e}")
        except Exception as e:
            print(f"위치 레이블 할당 중 오류 발생: {e}")
        return data


#모듈이기때문에 외부에서 모듈을 불러올때 아래 위치 dict넣어줌
# location_dict = {
#     (37.250536, 127.155263): 'hallway',
#     (37.250591, 127.155418): 'hallway',
#     (37.25062, 127.155515): 'hallway',
#     (37.250482, 127.155235): 'ward',
#     (37.250507, 127.155302): 'ward',
#     (37.250526, 127.155369): 'ward',
#     (37.250652, 127.15543): 'other',
#     (37.250672, 127.155485): 'other',
#     (37.250689, 127.155541): 'other',
#     (37.250572, 127.155225): 'ward',
#     (37.250595, 127.155294): 'ward',
#     (37.250608, 127.155335): 'ward',
#     (37.250546, 127.15544): 'ward',
#     (37.25051, 127.155479): 'ward',
#     (37.250576, 127.155568): 'other',
#     (37.250549, 127.155609): 'other',
#     (37.250561, 127.155194): 'ward',
#     (37.250493, 127.155269): 'ward',
#     (37.250584, 127.15526): 'ward',
#     (37.250516, 127.155336): 'ward',
#     (37.25054, 127.155402): 'ward',
#     (37.250662, 127.155458): 'other',
#     (37.250682, 127.155513): 'other',
#     (37.250562, 127.155333): 'hallway',
#     (37.250563, 127.155194): 'ward',
#     (37.250573, 127.15523): 'ward',
#     (37.250584, 127.155262): 'ward',
#     (37.250595, 127.155294): 'ward',
#     (37.250608, 127.155338): 'ward',
#     (37.250656, 127.155437): 'other',
#     (37.250667, 127.155466): 'other',
#     (37.250673, 127.155495): 'other',
#     (37.250682, 127.155521): 'other',
#     (37.250485, 127.155235): 'ward',
#     (37.250497, 127.155269): 'ward',
#     (37.250506, 127.155302): 'ward',
#     (37.250518, 127.155337): 'ward',
#     (37.250529, 127.15537): 'ward',
#     (37.25054, 127.155404): 'ward',
#     (37.250544, 127.155446): 'ward',
#     (37.250511, 127.155477): 'other',
#     (37.250571, 127.155542): 'other',
#     (37.250588, 127.155603): 'other',
#     (37.250551, 127.155606): 'other',
#     (37.250568, 127.155353): 'hallway',
#     (37.250602, 127.155453): 'hallway',
#     (37.250621, 127.155522): 'hallway'
# }
