import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml


@dataclass
class GoldIngestionConfig:
    '''Configuration for the gold layer data ingestion component'''
    silver_base_path: str = os.path.join('artifacts', 'silver')
    gold_base_path: str = os.path.join('artifacts', 'gold')

    def get_silver_partition_path(self, year: int, month: int, file_name: str) -> str:
        '''
        Generate silver partition path for reading

        Args:
            year: The year of the data
            month: The month of the data
            file_name: The name of the file

        Returns:
            str: Complete partition path
        '''
        partition_path = f"{year}/{month:02d}"
        return os.path.join(self.silver_base_path, partition_path, file_name)

    def get_gold_partition_path(self, year: int, month: int, file_name: str) -> str:
        '''
        Generate gold partition path for writing

        Args:
            year: The year of the data
            month: The month of the data
            file_name: The name of the file

        Returns:
            str: Complete partition path
        '''
        partition_path = f"{year}/{month:02d}"
        return os.path.join(self.gold_base_path, partition_path, file_name)


class GoldIngestion:
    '''
    Gold layer data ingestion component.
    Handles ML-ready feature engineering and selection
    '''
    def __init__(self, config_path: str = 'config/data_config.yaml'):
        '''
        Initialize gold layer data ingestion component

        Args:
            config_path: Path to the configuration file
        '''
        self.gold_ingestion_config = GoldIngestionConfig()
        self.params = read_yaml(config_path)['gold_ingestion']
        self.feature_selection_config = read_yaml(config_path)['feature_selection']
        logging.info(f"Gold layer data ingestion component initialized")

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Select final features for the gold layer

        Args:
            df: Dataframe for feature selection

        Returns:
            pd.DataFrame: Dataframe with selected features
        '''
        try:
            logging.info("Starting feature selection for ML dataset")

            # Get selected features from config
            selected_features = set(self.feature_selection_config)

            # Check for features availability
            existing_features = set(df.columns)
            available_features = selected_features & existing_features
            missing_features = selected_features - existing_features

            if missing_features:
                logging.warning(f"The dataset is missing the following features: {missing_features}")

            # Keep only available features
            features_to_keep = list(available_features)
            df = df[features_to_keep]
            
            logging.info(f"Selected {len(features_to_keep)} features for the gold layer")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def load_silver_partitions(self) -> list:
        '''
        Discover and load silver partitions for ingestion

        Returns:
            list: List of tuples (year, month) for available partitions
        '''
        try:
            logging.info("Discovering and loading silver partitions")
            partitions = []

            silver_base = self.gold_ingestion_config.silver_base_path
            if not os.path.exists(silver_base):
                raise FileNotFoundError(f"Silver base path not found: {silver_base}")

            # Get silver file name from config
            silver_file_name = self.params['silver_file_name']

            # Get all the partition files
            partition_files = Path(silver_base).glob(f'*/*/{silver_file_name}')

            # Extract year and month from the partition file path
            for partition_file in partition_files:
                try:
                    month_dir = partition_file.parent
                    year_dir = month_dir.parent

                    # Validate that parents are directories
                    if not month_dir.is_dir() or not year_dir.is_dir():
                        logging.warning(f"Parent paths are not directories for: {partition_file}")
                        continue

                    # Validate and convert directory names to integers
                    if month_dir.name.isdigit() and year_dir.name.isdigit():
                        month = int(month_dir.name)
                        year = int(year_dir.name)

                        if 1 <= month <= 12:
                            partitions.append((year, month))
                            logging.info(f"Found valid partition: {year}-{month:02d}")
                        else:
                            logging.warning(f"Invalid month value in partition: {partition_file}")
                    else:
                        logging.warning(f"Invalid directory names in partition: {partition_file}")
                
                except ValueError:
                    logging.warning(f"Invalid partition path structure: {partition_file}")
                    continue

            logging.info(f"Found {len(partitions)} available silver partitions")
            return sorted(partitions)

        except Exception as e:
            raise CustomException(e, sys)

    def save_gold_partitions(self, df: pd.DataFrame, year: int, month: int) -> str:
        '''
        Save transformed data to gold partition

        Args:
            df: Transformed dataframe
            year: Year for partition
            month: Month for partition

        Returns:
            str: Path to saved partition
        '''
        try:
            # Get gold file name from config
            gold_file_name = self.params['gold_file_name']

            # Generate partition path
            partition_path = self.gold_ingestion_config.get_gold_partition_path(year, month, gold_file_name)
            partition_dir = os.path.dirname(partition_path)     # Retrieve directory path without the file name

            # Create partition directory if it doesn't exist
            os.makedirs(partition_dir, exist_ok=True)

            # Save partition as parquet
            df.to_parquet(partition_path, index=False)

            logging.info(f"Saved gold partition {year}-{month:02d}: {len(df)} records to {partition_path}")
            return partition_path

        except Exception as e:
            raise CustomException(e, sys)
            
    def initiate_gold_ingestion(self) -> list:
        '''
        Initiate the gold layer data ingestion and feature selection process

        Returns:
            list: List of gold layer data partitions paths
        '''
        logging.info("Starting gold layer data ingestion and feature selection")
        try:
            # Discover silver partitions
            silver_partitions = self.load_silver_partitions()

            if not silver_partitions:
                raise ValueError("No silver partitions found")

            saved_paths = []

            # Get silver file name from config
            silver_file_name = self.params['silver_file_name']

            # Process each partition
            for year, month in silver_partitions:
                logging.info(f"Processing silver partition {year}-{month:02d}")

                # Load silver data
                silver_path = self.gold_ingestion_config.get_silver_partition_path(year, month, silver_file_name)
                df = pd.read_parquet(silver_path)
                logging.info(f"Loaded silver partition {year}-{month:02d}: {len(df)} records")
                
                # Apply feature selection
                df = self.select_features(df)
                logging.info(f"Applied feature selection to silver partition {year}-{month:02d}")

                # Save to gold partition
                gold_path = self.save_gold_partitions(df, year, month)
                saved_paths.append(gold_path)
                logging.info(f"Completed processing gold partition {year}-{month:02d} and saved to {gold_path}")

            logging.info(f"Gold layer data ingestion and feature selection completed successfully")
            logging.info(f"Total gold partitions processed: {len(saved_paths)}")
            return saved_paths

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    gold_ingestion = GoldIngestion()
    saved_paths = gold_ingestion.initiate_gold_ingestion()
    print(f"Gold partitions saved to: {saved_paths}")