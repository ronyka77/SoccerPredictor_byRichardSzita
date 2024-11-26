@echo off
echo Starting data processing sequence...

REM Set Python path - adjust this to your Python installation path
set PYTHON_PATH="\\192.168.0.77\Betting\Chatgpt\football_env\Scripts\python.exe"

echo Running fbref_get_data.py...
%PYTHON_PATH% fbref_get_data.py
if errorlevel 1 (
    echo Error running fbref_get_data.py
    pause
    exit /b 1
)
echo fbref_get_data.py completed successfully.

echo Running odds_scraper.py...
%PYTHON_PATH% odds_scraper.py
if errorlevel 1 (
    echo Error running fbref_scraper.py
    pause
    exit /b 1
)
echo odd_scraper.py completed successfully.

echo Running fbref_scraper.py...
%PYTHON_PATH% fbref_scraper.py
if errorlevel 1 (
    echo Error running fbref_scraper.py
    pause
    exit /b 1
)
echo fbref_scraper.py completed successfully.

echo Running merge_odds.py...
%PYTHON_PATH% merge_odds.py
if errorlevel 1 (
    echo Error running merge_odds.py
    pause
    exit /b 1
)
echo merge_odds.py completed successfully.

echo Running aggregation.py...
%PYTHON_PATH% aggregation.py
if errorlevel 1 (
    echo Error running aggregation.py
    pause
    exit /b 1
)
echo aggregation.py completed successfully.

echo All processes completed successfully!
pause