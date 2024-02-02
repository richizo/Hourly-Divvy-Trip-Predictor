import logging
from typing import Optional

def get_logger(name: Optional[str] = "divvy_trips") -> logging.Logger:
  
  logger = logging.getLogger(name)
  
  # If the logger with the specified name doesn't exist, create it with the specifications below.
  if not logger.handlers:
    
    logger.setLevel(logging.DEBUG)
    
    # Create a handler, its level, and its formatting protocol
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    
    # Set the formatter
    console_handler.setFormatter(formatter)
    
    # Associate this handler with the logger
    logger.addHandler(console_handler)
    
    return logger