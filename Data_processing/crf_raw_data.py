import pandas as pd

class DataProcessor:
    def __init__(self, location_data=None, sensor_data=None, crf_data=None, trait_data=None):
        """
        Constructor for the DataProcessor class.
        Data can be directly provided as DataFrames or loaded later using the load_data method.
        """
        self.location_data = location_data
        self.sensor_data = sensor_data
        self.crf_data = crf_data
        self.trait_data = trait_data
        self.merged_data = None

    def load_data(self, location_file=None, sensor_file=None, crf_file=None, trait_file=None):
        """
        Function to load CSV files and store them as DataFrames.
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
        Function to merge location data and sensor data on common columns.
        """
        if self.location_data is None or self.sensor_data is None:
            raise ValueError("Location data or Sensor data is missing.")
        
        # Ensure both dataframes have the same 'targetTime' dtype
        self.location_data['targetTime'] = pd.to_datetime(self.location_data['targetTime'])
        self.sensor_data['targetTime'] = pd.to_datetime(self.sensor_data['targetTime'])
        
        # Merge the location and sensor dataframes on 'key_id' and 'targetTime'
        self.merged_data = pd.merge(self.location_data, self.sensor_data, on=['key_id', 'targetTime'], how='inner')
        
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
        Function to process CRF data and merge it with merged_data.
        """
        if self.crf_data is None or self.merged_data is None:
            raise ValueError("CRF data or merged data is missing.")

        # Remove unnecessary columns and rename 'key_id'
        if 'Unnamed: 0' in self.crf_data.columns:
            self.crf_data = self.crf_data.drop(columns=['Unnamed: 0'])
        if 'researchNum' in self.crf_data.columns:
            self.crf_data = self.crf_data.drop(columns=['researchNum'])
        
        drop_columns = ['VE_tx', 'VE_tx_inj_time', 'VE_tx_secl_startime', 'VE_tx_secl_endtime', 
                        'VE_tx_rest_startime', 'VE_tx_rest_endtime', 'week', 'Eval_datetime', 'elapsed_date']
        existing_columns = [col for col in drop_columns if col are in self.crf_data.columns]
        if existing_columns:
            self.crf_data = self.crf_data.drop(columns=existing_columns)
        
        # Convert 'VE_tx_time' to datetime
        self.crf_data['VE_tx_time'] = pd.to_datetime(self.crf_data['VE_tx_time'])
        
        # Sort merged_data by 'key_id' and 'targetTime'
        self.merged_data.sort_values(by=['key_id', 'targetTime'], inplace=True)
        
        # Merge CRF data with merged_data
        for key, group in self.merged_data.groupby('key_id'):
            first_date = group['targetTime'].min()
            crf_subset = self.crf_data[self.crf_data['key_id'] == key].reset_index(drop=True)
            
            # Map data at 7-day intervals
            for i, crf_row in crf_subset.iterrows():
                start_date = first_date + pd.Timedelta(days=7 * i)
                end_date = start_date + pd.Timedelta(days=7)
                mask = (self.merged_data['key_id'] == key) & (self.merged_data['targetTime'] >= start_date) & (self.merged_data['targetTime'] < end_date)
                for col in crf_row.index:
                    if col not in ['key_id', 'targetTime']:
                        self.merged_data.loc[mask, col] = crf_row[col]
            
            # Handle emergency cases: map to the record creation date
            emergency_data = crf_subset[crf_subset['status'] == 'emergency']
            for emg_row in emergency_data.itertuples():
                date_match = (self.merged_data['key_id'] == key) & (self.merged_data['targetTime'].dt.date == getattr(emg_row, 'VE_tx_time').date())
                if date_match.any():
                    for col in crf_subset.columns.difference(['key_id', 'VE_tx_time', 'status']):
                        self.merged_data.loc[date_match, col] = getattr(emg_row, col)
        
        # Drop the 'idxInfo' column if it exists
        if 'idxInfo' in self.merged_data.columns:
            self.merged_data = self.merged_data.drop(columns=['idxInfo'])

    def merge_trait_data(self):
        """
        Function to merge trait data with the merged data.
        """
        if self.trait_data is None or self.merged_data is None:
            raise ValueError("Trait data or merged data is missing.")
        
        # Merge trait data with merged_data
        trait_subset = self.trait_data
        
        self.merged_data = pd.merge(self.merged_data, trait_subset, on='key_id')
        
        return self.merged_data
