import os
import sys
from src.logger import logging
from zipfile import Path
import numpy as np
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
    def export_df_from_db(self):
        pass
    def initiate_data_ingestion(self):
        try:
            logging.info("Initiating Data Ingestion!")
            df=self.export_df_from_db()
            logging.info("Dataset read from dataframe")
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            logging.info("Train Test Split")
            df.sort_values(by='my_last_updated', inplace=True)
            df_train = df.iloc[:int(df.shape[0]*0.80)]
            df_test = df.iloc[int(df.shape[0]*0.80):]
            logging.info(f"train shape:{df_train.shape}, test shape:{df_test.shape}")
            df_train.to_csv(self.data_ingestion_config.train_data_path,index=False)
            df_test.to_csv(self.data_ingestion_config.test_data_path,index=False)
            logging.info("Data ingestion completed")
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
     
        except Exception as e: raise CustomException(e,sys)