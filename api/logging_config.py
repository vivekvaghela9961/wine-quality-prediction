import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Set up a logger with a structured format writing to both console and rotating file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Define structured format
    log_format = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # File handler (rotates at 5MB, keeps 3 backups)
    file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    file_handler.setFormatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers if logger is already configured
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

# Initialize loggers
api_logger = setup_logger("api", "logs/api.log")
training_logger = setup_logger("training", "logs/training.log")
