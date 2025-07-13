import os
import sys

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, roc_auc_score, average_precision_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        '''
        This function is responsible for model training
        '''
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(),
                "CatBoost": CatBoostClassifier(verbose=False),
                "LightGBM": LGBMClassifier(verbose=-1)
            }

            params = {
                'Logistic Regression': {
                    'class_weight': 'balanced',
                    'C': np.logspace(-2, 1, 20),
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'solver': ['liblinear', 'saga', 'sag'],
                    'max_iter': np.arange(100, 1000, 100)
                },
                'Decision Tree': {
                    'class_weight': ['balanced', 'balanced_subsample'],
                    'max_depth': np.arange(2, 20, 2),
                    'min_samples_split': np.arange(2, 20, 2),
                    'min_samples_leaf': np.arange(1, 10, 1),
                    'max_features': ['sqrt', 'log2', None],
                    'criterion': ['gini', 'entropy', 'log_loss']
                },
                'Random Forest': {
                    'class_weight': ['balanced', 'balanced_subsample'],
                    'n_estimators': np.arange(100, 1000, 100),
                    'max_depth': [None] + list(np.arange(2, 20, 2)),
                    'min_samples_split': np.arange(2, 20, 2),
                    'min_samples_leaf': np.arange(1, 10, 1),
                    'max_features': ['sqrt', 'log2', None]
                },
                'XGBoost': {
                    'objective': ['binary:logistic'],
                    'scale_pos_weight': np.arange(1, 10, 2),
                    'eval_metric': ['auc', 'logloss', 'error'],
                    'n_estimators': np.arange(100, 1000, 100),
                    'learning_rate': np.logspace(-3, -1, 12),
                    'max_depth': np.arange(2, 20, 2),
                    'subsample': np.arange(0.5, 1, 0.1),
                    'colsample_bytree': np.arange(0.5, 1, 0.1),
                    'gamma': [0, 0.1, 0.2, 0.5, 1, 2, 5],
                    'alpha': np.logspace(-3, 1, 12),
                    'lambda': np.logspace(-2, 1, 12)
                },
                'LightGBM': {
                    'objective': 'binary',
                    'scale_pos_weight': np.arange(1, 10, 2),
                    'metric': ['auc', 'binary', 'average_precision'],
                    'n_estimators': np.arange(100, 1000, 100),
                    'learning_rate': np.logspace(-3, -1, 12),
                    'max_depth': np.arange(2, 20, 2),
                    'subsample': np.arange(0.5, 1, 0.1),
                    'colsample_bytree': np.arange(0.5, 1, 0.1),
                    'reg_alpha': np.logspace(-3, 1, 12),
                    'reg_lambda': np.logspace(-2, 1, 12)
                },
                'CatBoost': {
                    'objective': ['Logloss', 'CrossEntropy'],
                    'auto_class_weights': ['Balanced', 'SqrtBalanced'],
                    'eval_metric': ['AUC', 'Logloss', 'F1', 'Recall', 'Precision'],
                    'iterations': np.arange(100, 1000, 100),
                    'learning_rate': np.logspace(-3, -1, 12),
                    'depth': np.arange(2, 16, 2),
                    'l2_leaf_reg': np.arange(1, 10, 1),
                }
            }

            model_report:dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            # Set a threshold for the best model score
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and test dataset")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Get the best model's predictions
            predicted = best_model.predict(X_test)
            metrics = recall_score(y_test, predicted)   #TODO: Expand to other metrics

            return metrics
            
        except Exception as e:
            raise CustomException(e, sys)