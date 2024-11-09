import subprocess
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util_tools.logging_config import LoggerSetup

# Setup logging
logger = LoggerSetup.setup_logger(
    name='data_gathering',
    log_file='./data_tools/log/data_gathering.log',
    level=logging.INFO
)

def run_script(script_name):
    """Run a Python script and log the output."""
    try:
        # Check if script exists before running
        if not os.path.exists(script_name):
            logger.error(f"Script {script_name} not found")
            return

        logger.info(f"Running {script_name}...")
        # Use full path to Python executable
        python_path = sys.executable
        result = subprocess.run([python_path, script_name], 
                              capture_output=True, 
                              text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))) # Run from script directory
        
        if result.returncode == 0:
            logger.info(f"{script_name} completed successfully.")
            # Log stdout for debugging if needed
            if result.stdout:
                logger.debug(f"Output from {script_name}: {result.stdout}")
        else:
            logger.error(f"Error running {script_name}: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Exception occurred while running {script_name}: {str(e)}")
        raise  # Re-raise exception for proper error handling

def main():
    # Ensure log directory exists
    os.makedirs('./data_tools/log', exist_ok=True)
    
    scripts = [
        'fbref_get_data.py',
        'fbref_scraper.py', 
        'odds_scraper.py',
        'merge_odds.py',
        'aggregation.py'
    ]
    
    for script in scripts:
        try:
            run_script(script)
        except Exception as e:
            logger.critical(f"Critical error in script execution pipeline: {str(e)}")
            sys.exit(1)  # Exit if critical error occurs

if __name__ == "__main__":
    main()
