import logging
import os
from datetime import datetime

Log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(), "logs", Log_file)

os.makedirs(log_path, exist_ok=True)

Log_file_path = os.path.join(log_path, Log_file)

logging.basicConfig(
    filename=Log_file_path,
    level=logging.INFO,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s'
)
