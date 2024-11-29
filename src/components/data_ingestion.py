import os
import sys
from src.logger import logging
from zipfile import Path
import numpy as np
from src.exception import CustomException
import pandas as pd
import sqlite3
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','raw.csv')
    anime_images=os.path.join('artifacts','anime_images.csv')
    


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def export_df_from_db(self):
        try:
            # Connect to SQLite database
            logging.info("Connecting to the SQLite database...")
            conn = sqlite3.connect('anime_data.db')  # Adjust the path if necessary
            
            # Query to fetch all data from the 'anime_data' table
            query = "SELECT * FROM anime_data"
            
            # Load data into a Pandas DataFrame
            logging.info("Fetching data from the database...")
            df = pd.read_sql(query, conn)
            print(df.head())
            
            # Close the database connection
            conn.close()

            # Check if data was fetched
            if df is None or df.empty:
                logging.error("No data fetched from the database.")
                raise CustomException("No data fetched from the database", sys)

            logging.info(f"Data fetched successfully, shape of the data: {df.shape}")
            return df

        except Exception as e:
            logging.error(f"Error occurred while exporting data from DB: {str(e)}")
            raise CustomException(e, sys)


    def initiate_data_ingestion(self):
        try:
            logging.info("Initiating Data Ingestion!")
            df = pd.read_csv('/Users/nishchal_mac/Desktop/Data_Science/AnimeReccomendation/notebooks/merged_df.csv')  # Fetch the data from the database
            df['user_id'] = df['user_id'].astype('object')
            df['anime_id'] = df['anime_id'].astype('object')
            logging.info(f'data info: /n{df.info()}')
            if df is None or df.empty:
                logging.error("No data returned from export_df_from_db")
                raise CustomException("No data fetched from DB", sys)
            logging.info("Dataset read into DataFrame.")
            
            # Save the raw data
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved to {self.data_ingestion_config.raw_data_path}")
            
            # Split data into train and test sets
            logging.info("Train Test Split")
            df.sort_values(by='my_last_updated', inplace=True)  # Ensure the column exists in your data
            df_train = df.iloc[:int(df.shape[0] * 0.80)]  # 80% training data
            df_test = df.iloc[int(df.shape[0] * 0.80):]  # 20% test data
            
            logging.info(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")
            
            
            logging.info(f"sampling data begins!")
            df = pd.concat([df_train, df_test], ignore_index=True)

            # Finding all unique user IDs
            df_unique_user_id = pd.DataFrame()
            df_unique_user_id['user_id'] = np.unique(df['user_id'].values)

            # Sampling users randomly
            df_user_sample = df_unique_user_id.sample(n=10000, random_state=42)  # Added random_state for reproducibility

            # Creating a complete DataFrame for sampled users
            df_sample = pd.merge(df_user_sample, df, on='user_id')

            # Sorting the sample DataFrame with respect to 'my_last_updated'
            df_sample.sort_values(by='my_last_updated', inplace=True)

            # Splitting the sample data into train and test sample DataFrames
            df_train_sample = df_sample.iloc[:int(df_sample.shape[0] * 0.80)]
            df_test_sample = df_sample.iloc[int(df_sample.shape[0] * 0.80):]
            
            # Save train and test data
            df_train_sample.to_csv(self.data_ingestion_config.train_data_path, index=False)
            df_test_sample.to_csv(self.data_ingestion_config.test_data_path, index=False)
                        
            logging.info("Data ingestion completed successfully.")
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {str(e)}")
            raise CustomException(e, sys)
