import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class LocationProcessor:
    @staticmethod
    def load_data_from_csv(file_path):
        """
        Function to load and preprocess a CSV file.
        :param file_path: Path to the CSV file
        :return: Preprocessed DataFrame
        """

        try:
            data = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            print(f"CSV file {file_path} has been successfully loaded.")
            
            # Load and preprocess location data
            data = LocationProcessor.load_location_data(data)
            data = LocationProcessor.preprocess_location_data(data)
            data = LocationProcessor.resample_and_calculate(data)
            
            print("Data preprocessing is complete.")
            return data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            print("The file is empty.")
        except pd.errors.ParserError:
            print("An error occurred while parsing the file.")
        except Exception as e:
            print(f"An error occurred while loading the file: {e}")
        return None

    
    @staticmethod
    def load_location_data(data):
        """
        Function to load location data into a DataFrame.
        :param data: Original DataFrame containing location data
        :return: Preprocessed DataFrame
        """
        try:
            # Remove unnecessary columns
            columns_to_drop = [col for col in data.columns if '\t' in col]
            data = data.drop(columns=columns_to_drop)
            
            # Assign column names
            data.columns = ['No.', 'Device Name', 'key_id', 'Latitude', 'Longitude', 'targetTime']
            data = data.applymap(lambda x: x.strip(',') if isinstance(x, str) else x)
            
            # Remove unnecessary columns
            data = data.drop(columns=['No.', 'Device Name'])
            print("Location data has been successfully loaded.")
            return data
        except KeyError as e:
            print(f"Key error: {e}")
        except Exception as e:
            print(f"Error occurred while loading location data: {e}")
        return data
    
    @staticmethod
    def preprocess_location_data(data):
        """
        Function to preprocess location data.
        Converts 'targetTime' to datetime and sets the index.
        :param data: Location data DataFrame
        :return: Preprocessed DataFrame
        """
        try:
            data['targetTime'] = pd.to_datetime(data['targetTime'])
            data.set_index(['key_id', 'targetTime'], inplace=True)
            
            print("Location data preprocessing is complete.")
            return data
        except KeyError as e:
            print(f"Key error: {e}")
        except Exception as e:
            print(f"Error occurred during location data preprocessing: {e}")
        return data
    
    @staticmethod
    def calculate_mode(df):
        """
        Function to calculate the mode of the given DataFrame.
        :param df: DataFrame
        :return: Mode Series
        """
        modes = df.mode()
        if not modes.empty:
            return modes.iloc[0]
        else:
            return None
    
    @staticmethod
    def calculate_entropy(group):
        """
        Function to calculate entropy and normalized entropy within a group.
        :param group: DataFrame group
        :return: Entropy and normalized entropy (may be NaN)
        """
        if len(group) == 0:
            return np.nan, np.nan
        time_spent = group.groupby('key_id').size() / group.shape[0]
        entropy = -sum(time_spent * np.log(time_spent + 1e-10))  # Avoid log(0) by adding 1e-10
        max_entropy = np.log(len(group['key_id'].unique())) if group['key_id'].nunique() > 1 else 1
        normalized_entropy = entropy / max_entropy
        return entropy, normalized_entropy
    
    @staticmethod
    def calculate_location_variance(group):
        """
        Function to calculate the variance of latitude and longitude in the location data.
        :param group: DataFrame group
        :return: Logarithmic value of latitude and longitude variance
        """
        lat_var = np.var(group['Latitude'].astype(float))  # Convert strings to floats
        lon_var = np.var(group['Longitude'].astype(float))  # Convert strings to floats
        return np.log(lat_var + lon_var + 1e-10)  # Avoid log(0) by adding 1e-10
    
    @staticmethod
    def sliding_window_variability(df, window_size):
        variability = []
        for start in range(0, len(df) - window_size + 1, window_size):
            window = df.iloc[start:start + window_size]
            if len(window) > 1:
                var = LocationProcessor.calculate_location_variance(window)
                variability.append(var)
            else:
                variability.append(np.nan)

        return variability
    
    @staticmethod
    def resample_and_calculate(data):
        """
        Function to resample data and calculate entropy and location variability.
        :param data: Preprocessed location data DataFrame
        :return: DataFrame with resampled and calculated entropy and location variability
        """
        try:
            df_minute_mode = data.groupby(level='key_id').resample('1H', level='targetTime').apply(LocationProcessor.calculate_mode)
            df_minute_mode = df_minute_mode.reset_index()
            df_minute_mode['Date'] = df_minute_mode['targetTime'].dt.date
            
            # Calculate 24-hour entropy
            daily_entropy_df = data.reset_index()
            daily_entropy_df['Date'] = daily_entropy_df['targetTime'].dt.date
            daily_entropy_result = daily_entropy_df.groupby('Date').apply(LocationProcessor.calculate_entropy).apply(pd.Series).reset_index()
            daily_entropy_result.columns = ['Date', 'Daily_Entropy', 'Normalized_Daily_Entropy']
            df_minute_mode = df_minute_mode.merge(daily_entropy_result, on='Date', how='left')
            
            # Calculate 8-hour entropy
            eight_hour_entropy_df = data.reset_index()
            eight_hour_entropy_result = eight_hour_entropy_df.groupby(pd.Grouper(key='targetTime', freq='8H')).apply(LocationProcessor.calculate_entropy).apply(pd.Series).reset_index()
            eight_hour_entropy_result.columns = ['targetTime', 'Eight_Hour_Entropy', 'Normalized_Eight_Hour_Entropy']
            df_minute_mode = df_minute_mode.merge(eight_hour_entropy_result, on='targetTime', how='left')
            
            location_variability = LocationProcessor.sliding_window_variability(df_minute_mode, 8)
            df_minute_mode['Location_Variability'] = pd.Series(location_variability, index=df_minute_mode.index[:len(location_variability)])
            print("Resampling and entropy calculation is complete.")
            
            return df_minute_mode
        except Exception as e:
            print(f"Error occurred during resampling and calculation: {e}")
        return data
    
    @staticmethod
    def assign_location_labels(data, location_dict):
        """
        Function to assign location labels based on the closest location coordinates.
        :param data: Location data DataFrame
        :param location_dict: Dictionary mapping locations to coordinates
        :return: DataFrame with assigned location labels
        """
        try:
            data = data.fillna(0)
            # Convert 'Latitude' and 'Longitude' to numeric (coerce errors to NaN)
            data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
            data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')

            # Replace NaN with 0
            data = data.fillna(0)

            # Extract coordinate data from location_dict
            coords = np.array(list(location_dict.keys()))

            # Create and train NearestNeighbors model
            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(coords)

            def get_nearest_location(row):
                # Calculate the nearest location for each row's latitude and longitude
                distance, index = neigh.kneighbors([[row['Latitude'], row['Longitude']]])
                nearest_coord = coords[index[0][0]]
                return location_dict[tuple(nearest_coord)]

            # Map location names to the 'place' column
            data['place'] = data.apply(get_nearest_location, axis=1)

            # Drop 'Latitude' and 'Longitude' columns
            data = data.drop(columns=['Latitude', 'Longitude'])

            print("Location label assignment is complete.")
            return data
        except KeyError as e:
            print(f"Key error: {e}")
        except Exception as e:
            print(f"Error occurred during location label assignment: {e}")
        return data


# Module usage example:
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
