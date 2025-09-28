import os
import sys
from datetime import datetime
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml

@dataclass
class BronzeIngestionConfig:
    """Configuration for the bronze layer data ingestion component"""
    data_path: str = os.path.join('artifacts', 'bronze')
    file_name: str = 'raw_data.parquet'

    def get_partition_path(self, year: int, month: int) -> str:
        """
        Generate the full partition path for bronze data storage

        Args:
            year: The year of the data
            month: The month of the data

        Returns:
            str: Complete partition path
        """
        partition_path = f"{year}/{month:02d}"
        return os.path.join(self.data_path, partition_path, self.file_name)

class BronzeIngestion:
    """
    Bronze layer data ingestion component.
    Handles raw data ingestion with partitioning and minimal processing.
    """

    def __init__(self, config_path: str = 'config/data_config.yaml'):
        """
        Initialize bronze layer data ingestion component

        Args:
            config_path: Path to the configuration file
        """
        self.bronze_ingestion_config = BronzeIngestionConfig()
        self.params = read_yaml(config_path)['bronze_ingestion']
        logging.info(f"Bronze layer data ingestion component initialized")

    def prepare_disbursal_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DisbursalDate column to datetime format for partitioning
        
        Args:
            df: Dataframe with DisbursalDate column

        Returns:
            pd.DataFrame: Dataframe with DisbursalDate column converted to datetime format
        """
        try:
            logging.info("Converting DisbursalDate column to datetime format")

            # Convert to datetime format
            df['DisbursalDate'] = pd.to_datetime(df['DisbursalDate'], format='%d-%m-%y')

            logging.info("DisbursalDate column converted to datetime format")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def save_partitions(self, df: pd.DataFrame) -> list:
        """
        Save data partitioned by year and month

        Args:
            df: Dataframe with DisbursalDate column in datetime format

        Returns:
            list: List of saved partition paths
        """
        try:
            logging.info("Creating and saving partitions")
            saved_paths = []

            # Group by year and month
            df['year'] = df['DisbursalDate'].dt.year
            df['month'] = df['DisbursalDate'].dt.month
            
            # Create partitions
            for (year, month), df in df.groupby(['year', 'month']):
                # Remove temporary grouping columns
                partition_df = df.drop(['year', 'month'], axis=1)

                # Create partition directory if it doesn't exist
                partition_path = self.bronze_ingestion_config.get_partition_path(year, month)
                os.makedirs(partition_path, exist_ok=True)

                # Save partition as parquet
                partition_file = os.path.join(partition_path, self.bronze_ingestion_config.file_name)
                partition_df.to_parquet(partition_file, index=False)
                saved_paths.append(partition_file)
                logging.info(f"Saved partition {year}-{month:02d}: {len(partition_df)} records to {partition_file}")

            logging.info(f"Successfully created and saved {len(saved_paths)} partitions")
            return saved_paths

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_bronze_ingestion(self) -> list:
        """
        Initiate the bronze layer data ingestion process

        Returns:
            list: List of bronze layer data partitions paths
        """
        logging.info("Starting bronze layer data ingestion")
        try:
            # Load raw data
            source_path = self.params['source_path']
            df = pd.read_csv(source_path)
            logging.info("Read the dataset as dataframe")
            logging.info(f"Loaded data with shape: {df.shape}")

            # Convert DisbursalDate to datetime format
            df = self.prepare_disbursal_date(df)

            # Create and save partitions
            saved_paths = self.save_partitions(df)

            logging.info(f"Bronze layer data ingestion completed")
            return saved_paths

        except Exception as e:
            raise CustomException(e, sys)
