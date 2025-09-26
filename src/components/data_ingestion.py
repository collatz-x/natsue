import os
import sys

import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.parquet')
    val_data_path: str = os.path.join('artifacts', 'val.parquet')
    test_data_path: str = os.path.join('artifacts', 'test.parquet')
    raw_data_path: str = os.path.join('artifacts', 'data.parquet')
    
class DataIngestion:
    def __init__(self, config_path: str = 'config/data_config.yaml'):
        self.ingestion_config = DataIngestionConfig()
        self.params = read_yaml(config_path)['data_ingestion']
        
    def initiate_data_ingestion(self):
        '''
        This function is responsible for ingesting the data from the source
        '''
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(self.params['source_path'])                                             #TODO: change to the actual path of the data source
            logging.info('Read the dataset as dataframe')
            logging.info(f'Dataset shape: {df.shape}')

            # Extract the directory path
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Ingest and save the raw data to the artifacts folder
            df.to_parquet(self.ingestion_config.raw_data_path, index=False)
            logging.info(f'Raw data saved to {self.ingestion_config.raw_data_path}')

            # Train test split
            logging.info("Train test split initiated with stratification")
            logging.info(f'Stratifying on target column: {self.params["target_column"]}')

            # First split: 80% train_val, 20% test
            train_val_set, test_set = train_test_split(
                df,
                test_size=self.params['test_size'],
                random_state=self.params['random_state'],
                stratify=df[self.params['target_column']]
            )

            # Second split: 80% train, 20% val
            train_set, val_set = train_test_split(
                train_val_set,
                test_size=self.params['val_size'],
                random_state=self.params['random_state'],
                stratify=train_val_set[self.params['target_column']]
            )

            # Save the train, validation, and test data to the artifacts folder
            train_set.to_parquet(self.ingestion_config.train_data_path, index=False)
            val_set.to_parquet(self.ingestion_config.val_data_path, index=False)
            test_set.to_parquet(self.ingestion_config.test_data_path, index=False)

            logging.info("Ingestion of the data is completed")
            logging.info(f'Train set shape: {train_set.shape}')
            logging.info(f'Validation set shape: {val_set.shape}')
            logging.info(f'Test set shape: {test_set.shape}')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()