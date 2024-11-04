# Football Predictor by Richard Szita

This project is a comprehensive machine learning pipeline designed to predict football match outcomes. It leverages a combination of ensemble learning techniques and neural networks to enhance predictive accuracy.

## Features

- **Data Scraping and Storage**: Automated scripts in data_tools folder (`fbref_get_data.py` and `fbref_scraper.py` and `aggregation.py`) are used to scrape football data, which is then stored in a MongoDB database. The project includes the latest database backup for convenience.

- **Feature Engineering**: Features are engineered using two approaches:
  - Python scripts for initial processing.
  - Power BI files (located in the `data_tools\PowerBI` folder) for advanced feature engineering.

- **Model Training**: The project trains stacked models combining regression techniques, XGBoost, and Keras neural networks. Separate models are developed for predicting match outcomes, home goals, and away goals.

- **Prediction**: Scripts are provided to make predictions using the trained models.

## Installation

1. **Clone the Repository**:

   
   git clone <https://github.com/ronyka77/Football_predictor_byRichardSzita.git>

2. **Navigate to the Project Directory**:

   
   cd Football_predictor_byRichardSzita

3. **Create and Activate a Virtual Environment**:

   
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

4. **Install Dependencies**:

   
   pip install -r requirements.txt

5. **Install MongoDB Server**:

   Download and install MongoDB from the [official website](https://www.mongodb.com/try/download/community).

6. **Restore Database Backup**:

   Create a new database in MongoDB and restore the provided backup data.

## Usage

1. **Data Scraping**:

   Run the following scripts to fetch the latest data:

   
   python .\SoccerPredictor_byRichardSzita\data_tools\fbref_get_data.py
   python .\SoccerPredictor_byRichardSzita\data_tools\fbref_scraper.py
   python .\SoccerPredictor_byRichardSzita\data_tools\aggregation.py

2. **Feature Engineering**:

   - **Python-Based**: Execute the feature engineering Python script to prepare data for modeling and prediction.

   - **Power BI-Based**: Open the Power BI files located in the `data_tools` folder for model data (both prediction and training). Refresh the data and export the content of Worksheet1 to CSV files (`model_data_training_newPoisson.csv` for training and `model_data_prediction_newPoisson.csv` for prediction).

3. **Additional Data Processing**:

   - **Add ELO Scores**: Run the `add_ELO_scores.py` script to incorporate ELO scores into the dataset.

   - **Add xPoisson Columns**: Execute the `add_poisson_xG.py` script to add expected Poisson distribution columns.

   - **Merge Data**: Run the `merge_data_for_prediction.py` script to combine all processed data.

4. **Final Data Preparation**:

   Open the Power BI merge data file, refresh the data, and export it to a CSV file named `merged_data_prediction.csv`.

5. **Model Training**:

   Run the model training scripts in score_prediction folder to train the models on the prepared data. --> `model_stacked_2fit_outcome.py`, `model_stacked_2fit_homegoals.py`, `model_stacked_2fit_awaygoals.py`

6. **Prediction**:

   Use the prediction scripts to generate predictions based on the trained models. --> `stacked_2fit_outcome_prediction.py`, `stacked_2fit_homegoals_prediction.py`, `stacked_2fit_awaygoals_prediction.py`

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License.

## Acknowledgments

Special thanks to the open-source community for providing the tools and libraries that made this project possible.
