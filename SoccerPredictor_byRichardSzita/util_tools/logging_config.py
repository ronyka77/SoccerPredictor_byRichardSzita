import logging
import os
from typing import Optional

class LoggerSetup:
    @staticmethod
    def setup_logger(
        name: str,
        log_file: str,
        level: int = logging.INFO,
        format_string: Optional[str] = None
    ) -> logging.Logger:
        """Configure and return a logger instance"""
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(format_string)
        file_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)

        return logger 