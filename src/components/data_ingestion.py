import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml


@dataclass
class DataIngestionConfig:
    '''
    Configuration for the final data ingestion component
    '''
    processed_data_path: str = os.path.join('artifacts', 'processed')
    gold_base_path: str = os.path.join('artifacts', 'gold')

    def get_gold_partition_path(self, year: int, month: int, file_name: str) -> str:
        '''
        Generate gold partition path for reading
        
        Args:
            year: The year of the data
            month: The month of the data
            file_name: The name of the file

        Returns:
            str: Complete partition path
        '''
        partition_path = f"{year}/{month:02d}"
        return os.path.join(self.gold_base_path, partition_path, file_name)

    def get_processed_data_path(self, file_name: str) -> str:
        '''
        Generate processed data path
        
        Args:
            file_name: The name of the file

        Returns:
            str: Complete file path
        '''
        return os.path.join(self.processed_data_path, file_name)


class DataIngestion:
    '''
    Final data ingestion component.
    Shuffles and splits the data into train, validation, and test sets
    '''
    def __init__(self, config_path: str = 'config/data_config.yaml'):
        '''
        Initialize the final data ingestion component

        Args:
            config_path: Path to the configuration file
        '''
        self.ingestion_config = DataIngestionConfig()
        self.params = read_yaml(config_path)['data_ingestion']
        logging.info(f"Final data ingestion component initialized")

    def load_gold_partitions(self) -> pd.DataFrame:
        '''
        Discover and load gold partitions for ingestion

        Returns:
            pd.DataFrame: Concatenated dataframe
        '''
        try:
            logging.info("Discovering and loading gold partitions for train/test split")
            df_list = []

            gold_base = self.ingestion_config.gold_base_path
            if not os.path.exists(gold_base):
                raise FileNotFoundError(f"Gold base path not found: {gold_base}")

            # Get gold file name from config
            gold_file_name = self.params['gold_file_name']
            
            # Get all the partition files
            partition_files = Path(gold_base).glob(f'*/*/{gold_file_name}')

            # Load and concatenate all gold partitions
            for partition_file in partition_files:
                df_partition = pd.read_parquet(partition_file)
                df_list.append(df_partition)
                logging.info(f"Loaded gold partition {partition_file.parent.parent.name}-{partition_file.parent.name}: {len(df_partition)} records")

            # Concatenate all gold partitions
            df = pd.concat(df_list, ignore_index=True)
            logging.info(f"Concatenated {len(df_list)} gold partitions into a single dataframe")

            logging.info(f"Loaded final processed dataframe with shape: {df.shape}")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        '''
        Initialize the final data ingestion process and create train, validation, and test sets            
        '''
        logging.info("Starting final data ingestion and train/test split process")
        try:
            # Load consolidated gold dataframe
            df = self.load_gold_partitions()
            
            if df.empty:
                raise ValueError("No gold dataframe found")

            # Generate the directory path for final processed data
            file_path = self.ingestion_config.get_processed_data_path(self.params['full_file_name'])
            file_dir = os.path.dirname(file_path)

            # Generate the directory path for train, validation, and test data
            train_path = self.ingestion_config.get_processed_data_path(self.params['train_file_name'])
            val_path = self.ingestion_config.get_processed_data_path(self.params['val_file_name'])
            test_path = self.ingestion_config.get_processed_data_path(self.params['test_file_name'])

            # Create the processed data directory if it doesn't exist
            os.makedirs(file_dir, exist_ok=True)

            # Ingest and save the consolidated data
            df.to_parquet(file_path, index=False)
            logging.info(f'Consolidated data saved to {file_path}')

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
            train_set.to_parquet(train_path, index=False)
            val_set.to_parquet(val_path, index=False)
            test_set.to_parquet(test_path, index=False)

            logging.info("Final data ingestion and train/test split process completed")
            logging.info(f'Train set shape: {train_set.shape}')
            logging.info(f'Validation set shape: {val_set.shape}')
            logging.info(f'Test set shape: {test_set.shape}')
            return (
                train_path,
                val_path,
                test_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_path, val_path, test_path = data_ingestion.initiate_data_ingestion()
    print(f"Train set saved to: {train_path}")
    print(f"Validation set saved to: {val_path}")
    print(f"Test set saved to: {test_path}")