import os
import sys

import numpy as np
import pandas as pd
import pickle
import yaml
import re

from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, roc_auc_score, average_precision_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    '''
    This function is responsible for saving the model object

    Args:
        file_path: Path to save the model object
        obj: Model object to save
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Model object successfully saved at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    '''
    This function is responsible for evaluating the models

    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        models: Dictionary of model names and model objects
        params: Dictionary of model parameters

    Returns:
        report: Dictionary of model names and their corresponding metrics
    '''
    try:
        report = {}

        for name, model in models.items():
            # Set parameters
            para = params[name]
            rs = RandomizedSearchCV(
                estimator=model,
                param_distributions=para,
                n_iter=50,
                cv=5,
                verbose=0,
                n_jobs=-1,
                random_state=42
            )

            logging.info(f"RandomizedSearchCV initialized for {name}")

            # Fit the model for hyperparameter tuning using RandomizedSearchCV
            rs.fit(X_train, y_train)

            logging.info(f"RandomizedSearchCV completed for {name}")

            # Set best parameters to the model instance
            model.set_params(**rs.best_params_)

            logging.info(f"Best parameters set for {name}: {rs.best_params_}")
            logging.info(f"Model training using best parameters started for {name}")

            # Train the model on the full training set
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Get proba
            try:
                proba = model.predict_proba(X_test)
                y_pred_proba = proba[:, 1]              # Probability of positive class (i.e., "default" class)
            except Exception as e:
                print(f"Warning: predict_proba failed for {name}: {e}")
                y_pred_proba = y_pred.astype(float)     # Fallback to predicted classes
            
            # Evaluate the model
            metrics = recall_score(y_test, y_pred, average='binary')    #TODO: Expand to other metrics

            report[name] = metrics

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    

def read_yaml(file_path: str) -> dict:
    '''
    This function is responsible for reading the yaml file
    '''
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise CustomException(e, sys)


def load_model_params(config_path='config/model_params.yaml'):
    '''
    This function is responsible for loading the model parameters

    Args:
        config_path: Path to the model parameters configuration file

    Returns:
        models: Dictionary of model names and their corresponding parameters
    '''
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Process the parameters
        for _, model_params in config['models'].items():
            for param, values in model_params.items():
                if isinstance(values, dict) and 'type' in values:
                    if values['type'] == 'numpy.logspace':
                        model_params[param] = np.logspace(*values['args'])
                    elif values['type'] == 'numpy.arange':
                        model_params[param] = np.arange(*values['args'])
                    else:
                        raise ValueError(f"Unsupported parameter type: {values['type']}")
                    
        logging.info(f"Model hyperparameters successfully loaded from {config_path}")
        
        return config['models']

    except Exception as e:
        raise CustomException(e, sys)


def convert_to_datetime(df: pd.DataFrame, column: str | list[str], format: str=None, errors: str='coerce') -> pd.DataFrame:
    '''
    Convert a dataframe column to datetime format

    Args:
        df: Dataframe to modify
        column: Column to convert
        format: Datetime format string (optional)
        errors: Error handling strategy (optional)

    Returns:
        pd.DataFrame: Dataframe with the column converted to datetime format
        '''
    try:
        # Check if 'column' is a string or a list
        if isinstance(column, str):
            col_list = [column]
        elif isinstance(column, list):
            col_list = column
        else:
            raise TypeError(f"Column argument must be a string or a list of strings")
        
        for col in col_list:
            if format:
                df[col] = pd.to_datetime(df[col], format=format, errors=errors)
            else:
                df[col] = pd.to_datetime(df[col], errors=errors)

        logging.info(f"Converted {col_list} to datetime format")
        return df

    except Exception as e:
        raise CustomException(e, sys)


def convert_to_categorical(df: pd.DataFrame, column: str | list[str]) -> pd.DataFrame:
    '''
    Convert a dataframe column to categorical format

    Args:
        df: Dataframe to modify
        column: Column to convert

    Returns:
        pd.DataFrame: Dataframe with the column converted to categorical format
        '''
    try:
        # Check if 'column' is a string or a list
        if isinstance(column, str):
            col_list = [column]
        elif isinstance(column, list):
            col_list = column
        else:
            raise TypeError(f"Column argument must be a string or a list of strings")
            
        for col in col_list:
            df[col] = df[col].astype(str)

        logging.info(f"Converted {col_list} to categorical format")
        return df

    except Exception as e:
        raise CustomException(e, sys)


def convert_to_months(time_str: str) -> int | None:
    '''
    Convert `Xyrs Ymon` format string to number of months

    Args:
        time_str: String in `Xyrs Ymon` format

    Returns:
        int: Total number of months, or None if parsing fails
    '''
    try:
        if pd.isna(time_str):
            return None
            
        # Extract years and months
        match = re.match(r'(\d+)\s*yrs\s*(\d+)\s*mons?', time_str)
        if match:
            years = int(match.group(1))
            months = int(match.group(2))
            return years * 12 + months
        
        logging.info(f"Converted time duration string values to integer number of months")
        return None
    
    except Exception as e:
        raise CustomException(e, sys)
            

def calculate_ratio(df: pd.DataFrame, numerator_col: str, denominator_col: str, new_col: str) -> pd.DataFrame:
    '''
    Calculate the ratio of two columns with handling of division by zero

    Args:
        df: pandas DataFrame to modify
        numerator_col: Column to use as numerator
        denominator_col: Column to use as denominator
        new_col: Name of the new ratio column

    Returns:
        pd.DataFrame: DataFrame with the new ratio column added
    '''
    try:
        df[new_col] = np.where(
            df[denominator_col] > 0,
            df[numerator_col] / df[denominator_col],
            df[numerator_col]       # If denominator is 0, use numerator as ratio
        )

        logging.info(f"Calculated ratio for {new_col}")
        return df

    except Exception as e:
        raise CustomException(e, sys)