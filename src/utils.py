import os
import sys
import dill
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.loggers import logging
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        """
        Save an object to a file using dill.
        """
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        logging.info(f"Object saved at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate the performance of different regression models and return a report.
    """
    try:
        model_report = {}
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = models[model_name]
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            model_report[model_name] = {'train_score': train_model_score, 'test_score': test_model_score}
            logging.info(f"{model_name} Train R2 Score: {train_model_score}")
            logging.info(f"{model_name} Test R2 Score: {test_model_score}")
        
        return model_report

    except Exception as e:
        raise CustomException(e, sys)
    