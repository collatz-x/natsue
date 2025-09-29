import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml, convert_to_datetime, convert_to_categorical, convert_to_months, calculate_ratio


@dataclass
class SilverIngestionConfig:
    '''Configuration for the silver layer data ingestion component'''
    bronze_base_path: str = os.path.join('artifacts', 'bronze')
    silver_base_path: str = os.path.join('artifacts', 'silver')

    def get_bronze_partition_path(self, year: int, month: int, file_name: str) -> str:
        '''
        Generate bronze partition path for reading
        
        Args:
            year: The year of the data
            month: The month of the data
            file_name: The name of the file

        Returns:
            str: Complete partition path
        '''
        partition_path = f"{year}/{month:02d}"
        return os.path.join(self.bronze_base_path, partition_path, file_name)

    def get_silver_partition_path(self, year: int, month: int, file_name: str) -> str:
        '''
        Generate silver partition path for writing

        Args:
            year: The year of the data
            month: The month of the data
            file_name: The name of the file

        Returns:
            str: Complete partition path
        '''
        partition_path = f"{year}/{month:02d}"
        return os.path.join(self.silver_base_path, partition_path, file_name)


class SilverIngestion:
    '''
    Silver layer data ingestion component.
    Handles data cleaning, type standardization, and business feature engineering.
    '''
    def __init__(self, config_path: str = 'config/data_config.yaml'):
        '''
        Initialize silver layer data ingestion component

        Args:
            config_path: Path to the configuration file
        '''
        self.silver_ingestion_config = SilverIngestionConfig()
        self.params = read_yaml(config_path)['silver_ingestion']
        logging.info(f"Silver layer data ingestion component initialized")

    def standardize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Standardize data types for various columns in the silver layer

        Args:
            df: Dataframe to transform

        Returns:
            pd.DataFrame: Dataframe with standardized data types
        '''
        try:
            logging.info("Starting data type standardization")

            # Convert `Date.of.Birth` to datetime format
            df = convert_to_datetime(df, 'Date.of.Birth', format='%d-%m-%y')

            # Convert `manufacturer_id` to categorical format
            df = convert_to_categorical(df, 'manufacturer_id')

            # Convert time duration columns with string values to integer values representing number of months
            logging.info("Starting time duration column conversion")
            df['AVERAGE.ACCT.AGE'] = df['AVERAGE.ACCT.AGE'].apply(convert_to_months)
            df['CREDIT.HISTORY.LENGTH'] = df['CREDIT.HISTORY.LENGTH'].apply(convert_to_months)
            logging.info("All time duration columns conversion completed")

            logging.info("Data type standardization completed")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def clean_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Handle missing values in the dataset

        Args:
            df: Dataframe to clean

        Returns:
            pd.DataFrame: Dataframe with missing values handled
        '''
        try:
            logging.info("Starting missing value cleaning")

            # Impute missing values in `Employment.Type` to `Unemployed`
            missing_count = df['Employment.Type'].isna().sum()
            if missing_count > 0:
                logging.info(f"Counted {missing_count} missing values in `Employment.Type`")
                df['Employment.Type'] = df['Employment.Type'].fillna('Unemployed')
                logging.info(f"Imputed {missing_count} missing values in `Employment.Type` to `Unemployed`")
            else:
                logging.info("No missing values found in `Employment.Type`")

            logging.info("Missing value cleaning completed")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def create_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Create business logic features

        Args:
            df: Dataframe to feature engineer

        Returns:
            pd.DataFrame: Dataframe with new business logic features engineered
        '''
        try:
            logging.info("Starting business logic feature engineering")

            # Calculate age at disbursement
            df['age'] = df['DisbursalDate'].dt.year - df['Date.of.Birth'].dt.year
            logging.info("Created age feature (age at disbursement)")

            # Calculate credit utilization ratio
            df = calculate_ratio(df, 'PRI.DISBURSED.AMOUNT', 'PRI.SANCTIONED.AMOUNT', 'credit_utilization')
            logging.info("Created credit utilization ratio feature")

            # Calculate loan default ratio
            df = calculate_ratio(df, 'PRI.OVERDUE.ACCTS', 'PRI.ACTIVE.ACCTS', 'default_ratio')
            logging.info("Created loan default ratio feature")

            logging.info("Business logic feature engineering completed")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def load_bronze_partitions(self) -> list:
        '''
        Discover and load bronze partitions for ingestion

        Returns:
            list: List of tuples (year, month) for available partitions
        '''
        try:
            logging.info("Discovering and loading bronze partitions")
            partitions = []

            bronze_base = self.silver_ingestion_config.bronze_base_path
            if not os.path.exists(bronze_base):
                raise FileNotFoundError(f"Bronze base path not found: {bronze_base}")

            # Get bronze file name from config
            bronze_file_name = self.params['bronze_file_name']

            # Get all the partition files
            partition_files = Path(bronze_base).glob(f'*/*/{bronze_file_name}')

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

            logging.info(f"Found {len(partitions)} available bronze partitions")
            return sorted(partitions)

        except Exception as e:
            raise CustomException(e, sys)


    def save_silver_partitions(self, df: pd.DataFrame, year: int, month: int) -> str:
        '''
        Save transformed data to silver partition

        Args:
            df: Transformed dataframe
            year: Year for partition
            month: Month for partition

        Returns:
            str: Path to saved partition
        '''
        try:
            # Get silver file name from config
            silver_file_name = self.params['silver_file_name']
            
            # Generate partition path
            partition_path = self.silver_ingestion_config.get_silver_partition_path(year, month, silver_file_name)
            partition_dir = os.path.dirname(partition_path)     # Retrieve directory path without the file name

            # Create partition directory if it doesn't exist
            os.makedirs(partition_dir, exist_ok=True)

            # Save partition as parquet
            df.to_parquet(partition_path, index=False)
            
            logging.info(f"Saved silver partition {year}-{month:02d}: {len(df)} records to {partition_path}")
            return partition_path

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_silver_ingestion(self) -> list:
        '''
        Initiate the silver layer data ingestion and transformation process

        Returns:
            list: List of silver layer data partitions paths
        '''
        logging.info("Starting silver layer data ingestion and transformation")
        try:
            # Discover bronze partitions
            bronze_partitions = self.load_bronze_partitions()

            if not bronze_partitions:
                raise ValueError("No bronze partitions found")

            saved_paths = []

            # Get bronze file name from config
            bronze_file_name = self.params['bronze_file_name']

            # Process each partition
            for year, month in bronze_partitions:
                logging.info(f"Processing bronze partition {year}-{month:02d}")

                # Load bronze data
                bronze_path = self.silver_ingestion_config.get_bronze_partition_path(year, month, bronze_file_name)
                df = pd.read_parquet(bronze_path)
                logging.info(f"Loaded bronze partition {year}-{month:02d}: {len(df)} records")

                # Apply transformations
                df = self.standardize_data_types(df)
                df = self.clean_missing_values(df)
                df = self.create_business_features(df)
                logging.info(f"Applied transformations to bronze partition {year}-{month:02d}")

                # Save to silver partition
                silver_path = self.save_silver_partitions(df, year, month)
                saved_paths.append(silver_path)
                logging.info(f"Completed processing silver partition {year}-{month:02d} and saved to {silver_path}")

            logging.info(f"Silver layer data ingestion and transformation completed successfully")
            logging.info(f"Total silver partitions processed: {len(saved_paths)}")
            return saved_paths

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    silver_ingestion = SilverIngestion()
    saved_paths = silver_ingestion.initiate_silver_ingestion()
    print(f"Silver partitions saved to: {saved_paths}")