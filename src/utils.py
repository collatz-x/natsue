import os
import sys

import numpy as np
import pandas as pd
import pickle
import yaml

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

            # Fit the model for hyperparameter tuning using RandomizedSearchCV
            rs.fit(X_train, y_train)

            # Set best parameters to the model instance
            model.set_params(**rs.best_params_)

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