import pandas as pd
import numpy as np
from pandas import json_normalize
import json

class SensorDataProcessor:
    
    @staticmethod
    def load_sensing_data(file_path, chunksize=10000):
        """
        Function to load and process sensor data from a DataFrame.
        
        Parameters:
        file_path (str): The path to the sensor data file.
        chunksize (int): The size of chunks to split the data into.
        
        Returns:
        pd.DataFrame: The processed DataFrame.
        """
        # Create an empty list to store processed dataframes in chunks.
        chunks = []
        
        # Read and process the data in chunks.
        for chunk in pd.read_csv(file_path, sep='\t', encoding='utf-8', chunksize=chunksize):
            # Define the required columns
            chunk.columns = ['targetId', 'key_id', 'deviceId', 'data', 'targetTime']
            
            # Remove unnecessary characters and process JSON strings
            chunk['data'] = chunk['data'].str.replace('""', '"').str.replace('",', ',').str.strip(',')
            chunk = chunk.applymap(lambda x: x.strip(',') if isinstance(x, str) else x)

            # Normalize the JSON data in the 'data' column
            def normalize_json(x):
                try:
                    return json.loads(x) if not pd.isna(x) else {}
                except json.JSONDecodeError:
                    return {}
            
            df_normalized = json_normalize(chunk['data'].apply(normalize_json))
            chunk = pd.concat([chunk.reset_index(drop=True), df_normalized], axis=1).drop(columns=['data'])
            
            # Convert 'targetTime' to datetime format
            chunk['targetTime'] = pd.to_datetime(chunk['targetTime'], errors='coerce')  # Convert errors to NaT
            
            chunks.append(chunk)
        
        # Combine all the chunks into one DataFrame
        df = pd.concat(chunks, ignore_index=True)
        
        return df
    
    @staticmethod
    def process_sensing_data(combined_df):
        """
        Function to process sensor data by calculating total acceleration and removing unnecessary columns.
        
        Parameters:
        combined_df (pd.DataFrame): The input DataFrame.
        
        Returns:
        pd.DataFrame: DataFrame with total acceleration calculated and unnecessary columns removed.
        """
        # Check if all required columns exist
        required_columns = ['ACCELER_X_AXIS', 'ACCELER_Y_AXIS', 'ACCELER_Z_AXIS']
        missing_columns = [col for col in required_columns if col not in combined_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if all(column in combined_df.columns for column in required_columns):
            # Calculate total acceleration
            combined_df['TOTAL_ACCELERATION'] = np.sqrt(
                combined_df['ACCELER_X_AXIS']**2 + 
                combined_df['ACCELER_Y_AXIS']**2 + 
                combined_df['ACCELER_Z_AXIS']**2
            )
        
        # Define the columns to drop
        columns_to_drop = ['deviceId', 'ANGULAR_Y_AXIS', 'ANGULAR_X_AXIS', 'ANGULAR_Z_AXIS']
        
        # Drop only the columns that exist in the DataFrame
        existing_columns_to_drop = [col for col in columns_to_drop if col in combined_df.columns]
        combined_df = combined_df.drop(columns=existing_columns_to_drop)
        
        return combined_df


    @staticmethod
    def aggregate_sensing_data(data):
        """
        Function to aggregate sensor data in 1-hour intervals.
        
        Parameters:
        data (pd.DataFrame): The DataFrame containing processed sensor data.
        
        Returns:
        pd.DataFrame: Aggregated DataFrame with calculated statistics.
        """
        # Function to calculate the delta (difference between the first and last value)
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
        
        # Convert 'targetTime' to datetime and set it as index
        data['targetTime'] = pd.to_datetime(data['targetTime'])
        data.set_index('targetTime', inplace=True)
        
        # Resample the data in 1-hour intervals and apply the aggregation dictionary
        result = data.groupby('key_id').resample('1H').agg(agg_dict).reset_index()
        
        return result
    
    @staticmethod
    def reorganize_column_names(df):
        """
        Function to flatten multi-level column names and reorder them.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame.
        
        Returns:
        pd.DataFrame: DataFrame with reorganized column names.
        """
        # Flatten the multi-level column names
        df.columns = ['_'.join(col).strip() for col in df.columns.ravel()]

        # Reorder the column names
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
        
        # Rename 'key_id' and 'targetTime' columns
        df = df.rename(columns={'_key_id': 'key_id', '_targetTime': 'targetTime'})
        
        return df

"""
# Example usage:
# result = reorganize_column_names(result)
# print(result)

# Example usage:

# Assuming the DataFrame `df` contains the data:
df = pd.read_csv('/mount/nas/disk02/Data/Health/Mental_Health/SAFER/20240812/trait_state/seoul_first_half.csv', encoding='utf-8')
print(df)
# Load and process sensor data
sensing_data = SensorDataProcessor.load_sensing_data(df)
print(sensing_data)
sensing_data = SensorDataProcessor.process_sensing_data(sensing_data)

# Aggregate the processed sensor data
aggregated_data = SensorDataProcessor.aggregate_sensing_data(sensing_data)
aggregated_data = SensorDataProcessor.reorganize_column_names(aggregated_data)

# Optionally, save the final sensor data to CSV
aggregated_data.to_csv('processed_sensing_data.csv', index=False)

# Print the processed and aggregated sensor data
print(aggregated_data)
"""
