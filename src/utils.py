import os
import sys
import dill
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.loggers import logging

def save_object(file_path, obj):
    try:
        """
        Save an object to a file using dill.
        """
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        logging.info(f"Object saved at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)
    logging.info(f"Object saved at {file_path}")
