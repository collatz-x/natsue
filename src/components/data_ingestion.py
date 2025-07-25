import os
import sys

import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        '''
        This function is responsible for ingesting the data from the source
        '''
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/train.csv')                                             #TODO: change to the actual path of the data source
            logging.info('Read the dataset as dataframe')

            # Extract the directory path
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Ingest and save the raw data to the artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Train test split
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test data to the artifacts folder
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
