# Soccer Predictor by Richard Szita

![Football Analytics](https://img.shields.io/badge/Football-Analytics-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🏆 Project Overview

The Soccer Predictor by Richard Szita is an advanced machine learning pipeline designed to forecast football match outcomes with high accuracy. By leveraging a combination of ensemble learning techniques and neural networks, the system integrates data collection, feature engineering, and prediction models to deliver precise match predictions. Key functionalities include automated data scraping, sophisticated feature engineering, and a state-of-the-art stacked ensemble model that predicts match results, home goals, and away goals. This project is built with modern data science practices and a scalable architecture, making it a powerful tool for football analytics enthusiasts and professionals alike.

## 🎯 System Overview

The system consists of three powerful, integrated components:

1. 📊 Data Collection & Processing - Automated scraping and preprocessing
2. 🔧 Feature Engineering - Advanced statistical modeling
3. 🤖 Prediction Models - State-of-the-art ML ensemble

## 🏗️ Model Architecture

The Soccer Predictor employs a sophisticated stacked ensemble model to predict football match outcomes. The architecture is designed to leverage the strengths of various machine learning algorithms, ensuring robust and accurate predictions. Below are the key components and innovations of the model:

### Ensemble Components

The model utilizes a stacked ensemble approach, combining the predictive power of several machine learning algorithms:

- **LightGBM**: Known for its efficiency and speed, LightGBM is used for handling large datasets and providing high accuracy.
- **XGBoost**: A powerful gradient boosting framework that excels in predictive performance, particularly for structured data.
- **Neural Networks**: Implemented using Keras, the neural network component is tailored for capturing complex patterns in the data.
- **Random Forest**: Provides robustness and reduces overfitting by averaging multiple decision trees.
- **AdaBoost**: Enhances the model's performance by focusing on difficult-to-predict instances.

### Neural Network Architecture

The neural network is constructed using Keras with the following architecture:

- **Layers**: Includes Dense, Dropout, BatchNormalization, and Activation layers.
- **Activation Functions**: Utilizes ReLU for hidden layers and linear activation for the output layer.
- **Regularization**: L2 regularization is applied to prevent overfitting.
- **Callbacks**: Incorporates EarlyStopping and a custom learning rate scheduler to optimize training.

### Stacking Methodology

The ensemble models are combined using a stacking regressor, with a Ridge regression model serving as the meta-learner. This approach allows the model to learn from the predictions of individual models, improving overall accuracy.

### Feature Engineering

Feature engineering is a critical component of the model, involving:

- **Polynomial Features**: Generated to capture non-linear relationships in the data.
- **Recursive Feature Elimination (RFE)**: Used for selecting the most relevant features, enhancing model performance.

### Data Flow

The data processing pipeline includes several stages:

1. **Data Gathering**: Automated scripts collect data from various sources.
2. **Odds Scraping**: Retrieves betting odds to incorporate into the model.
3. **Data Merging and Aggregation**: Combines and processes data for feature engineering.
4. **Feature Engineering**: Enhances data with additional features for improved predictions.
5. **Model Training**: Trains the ensemble model using historical data.
6. **Prediction**: Generates predictions for upcoming matches.

### Innovations

The model benefits from the domain knowledge of a late football player, incorporating insights into the sport's dynamics. This expertise is reflected in the feature engineering and model design, providing a unique edge in predicting match outcomes.

This architecture ensures that the Soccer Predictor is both powerful and flexible, capable of delivering high-accuracy predictions for football matches.

## ✨ Features

### 📥 Data Collection & Storage

- Automated scraping from multiple sources:
  - FBRef match data and statistics (comprehensive match-level data)
  - Team performance metrics
  - Historical match data 
  - Betting odds data (for incorporating market insights)
- MongoDB integration for efficient data storage
- Backup and restore system included with data validation

### 🛠️ Feature Engineering

Two-tiered approach for maximum accuracy:

1. Python-based processing:
   
   - Cumulative statistics tracking:
     - Season-to-date goal sums
     - Running point totals
     
     - Progressive performance indicators
   - Team form indicators using weighted metrics
   - ELO rating integration with dynamic K-factor
   - Expected goals (xG) modeling using Poisson regression

2. Power BI advanced processing:
   - Complex feature derivation with DAX language
   - Moving averages for key metrics
   - Rolling averages calculation
   - Export capabilities for model training

### 🎯 Prediction Models

State-of-the-art stacked ensemble combining:
- LightGBM (gradient boosting)
- CatBoost (gradient boosting)
- XGBoost (gradient boosting)
- Neural Networks (Keras with custom architecture)
- Random Forest (ensemble learning)
- Separate models optimized for:
  - Match outcomes (W/D/L)
  - Home goals prediction
  - Away goals prediction

## 🚀 Installation

1. **Clone the Repository**:

git clone <https://github.com/ronyka77/Football_predictor_byRichardSzita.git>
cd Football_predictor_byRichardSzita

2. **Environment Setup**:
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt

3. **MongoDB Setup**:

- Install MongoDB Server from [official website](https://www.mongodb.com/try/download/community)
- Create new database called "football_data"
- Restore provided backup data (data/mongodb_backup/football_data)

## 🚀 Usage Guide

### 1. Data Collection

Run in sequence:

   python ./data_tools/fbref_get_data.py

   python ./data_tools/fbref_scraper.py

   python ./data_tools/odds_scraper.py

   python ./data_tools/merge_odds.py

   python ./data_tools/aggregation.py

### 2. Feature Engineering

#### Python Processing

Add additional features

   python ./data_tools/feature_engineering_for_predictions.py

   python ./data_tools/feature_engineering_for_model.py

   python ./data_tools/merge_data_for_prediction.py

#### Power BI Processing for Training Data

1. Open files(model_data_pred.pbix and model_data_training.pbix) in `data_tools/PowerBI/`
2. Refresh data
3. Export to CSV:
   - Training: `model_data_training.csv`
   - Prediction: `model_data_prediction.csv`

Add Poisson_xG and ELO scores

   python ./data_tools/add_poisson_xG.py

   python ./data_tools/add_ELO_scores.py

#### Power BI Processing for Predictions

1. Open merge_data_prediction.pbix in `data_tools/PowerBI/`
2. Refresh data
3. Export to CSV: `merged_data_prediction.csv`

Refresh Poisson_xG and ELO scores

   python ./data_tools/add_poisson_xG.py

   python ./data_tools/add_ELO_scores.py

### 3. Model Training
Run feature selection script:

   python ./Prediction_models/Feature_selector_for_model.py

This script performs feature selection for each model type (match outcome, home goals, away goals):
- Generates polynomial features
- Scales the data
- Uses recursive feature elimination (RFE) to select optimal features
- Saves selected features and scaler for each model type

Run the following scripts in `score_prediction` folder to train the prediction models:

   python model_stacked_2fit_outcome.py     # Trains stacked model for match outcome prediction (win/draw/loss)
                                           # Uses XGBoost, CatBoost and LightGBM as base models
                                           # with logistic regression as meta-learner

   python model_stacked_2fit_homegoals.py   # Trains stacked model for home team goals prediction
                                           # Uses XGBoost, CatBoost and LightGBM as base models 
                                           # with linear regression as meta-learner

   python model_stacked_2fit_awaygoals.py   # Trains stacked model for away team goals prediction
                                           # Uses XGBoost, CatBoost and LightGBM as base models
                                           # with linear regression as meta-learner

Each script:
- Loads the preprocessed training data
- Trains base models using cross-validation
- Generates meta-features through predictions
- Trains meta-learner on the stacked predictions
- Saves trained models for later prediction

### 4. Making Predictions

Execute prediction scripts:

   python predict_match_outcome.py

   python predict_home_goals.py

   python predict_away_goals.py

## 📁 Project Structure

Football_predictor_byRichardSzita/

├── data_tools/
│   ├── fbref_scraper.py
│   ├── fbref_get_data.py
│   ├── aggregation.py
│   ├── add_ELO_scores.py
│   ├── add_poisson_xG.py
│   ├── merge_data_for_prediction.py
│   ├── merge_odds.py
│   ├── feature_engineering_for_model.py
│   ├── feature_engineering_for_predictions.py
│   └── PowerBI/
│       ├── model_data_pred.pbix
│       ├── model_data_training.pbix
│       ├── merge_data_prediction.pbix
│       └── score_prediction/
│           ├── model_stacked_2fit_outcome.py
│           ├── model_stacked_2fit_homegoals.py
│           ├── model_stacked_2fit_awaygoals.py
│           ├── predict_match_outcome.py
│           ├── predict_home_goals.py
│           └── predict_away_goals.py
├── requirements.txt
└── README.md

## 📄 Requirements

Key dependencies:

- pandas
- numpy
- scikit-learn
- tensorflow
- xgboost
- pymongo
- beautifulsoup4
- Power BI Desktop

See `requirements.txt` for complete list.

## 📤 Output

The system generates:

- Predicted match outcomes
- Goal predictions
- Probability distributions
- Feature importance analysis

## 🤖 AI Models Utilized

This project was developed with the assistance of advanced AI models, which significantly contributed to the coding and development process. The AI models used include:

- **Cursor**: Assisted in code generation, debugging, and providing coding suggestions to improve efficiency and accuracy.
- **ChatGPT**: Provided natural language processing capabilities, helped in generating documentation, and offered insights and recommendations for coding best practices.

## 🎉 Acknowledgments

- Thanks to the open-source community for tools and libraries
- FBRef for providing comprehensive football statistics
- OddsPortal for providing betting odds data
- Contributors and maintainers of key dependencies

## 📊 Usage Examples

The output predictions generated by the system are provided in Excel format. These predictions can be utilized in various ways, including testing betting strategies and participating in friendly match tipping leagues. Below are some examples of how to interpret and use the output data:

### Example 1: Betting Strategy Testing

1. **Load the Predictions**:
   Open the Excel file containing the predicted match outcomes and goal predictions.

2. **Analyze the Predictions**:
   Review the predicted probabilities for match outcomes (Win/Draw/Loss) and the expected number of goals for both home and away teams.

3. **Develop a Betting Strategy**:
   Based on the predictions, create a betting strategy. For example, you might decide to place bets on matches where the model predicts a high probability of a home win.

4. **Test the Strategy**:
   Track the performance of your betting strategy over a series of matches. Compare the predicted outcomes with the actual results to evaluate the effectiveness of your strategy.

### Example 2: Friendly Match Tipping Leagues

1. **Load the Predictions**:
   Open the Excel file containing the predicted match outcomes and goal predictions.

2. **Make Your Predictions**:
   Use the model's predictions to make your own predictions for upcoming matches in a friendly tipping league. For example, if the model predicts a high probability of a draw, you might choose to tip a draw for that match.

3. **Track Your Performance**:
   Compare your predictions with the actual match results and track your performance in the tipping league. Use the insights gained from the model to improve your predictions over time.

### Example 3: Decision-Making for Team Management

1. **Load the Predictions**:
   Open the Excel file containing the predicted match outcomes and goal predictions.

2. **Analyze Team Performance**:
   Use the predicted probabilities and expected goals to assess the performance of your team and upcoming opponents. Identify strengths and weaknesses based on the model's predictions.

3. **Make Informed Decisions**:
   Use the insights from the predictions to make informed decisions about team strategy, player selection, and match preparation. For example, if the model predicts a high probability of conceding goals, you might focus on defensive training.

By leveraging the output predictions, you can make data-driven decisions and enhance your understanding of football match dynamics.

## 🔮 Future Work

There are several potential improvements and future directions for this project:

1. **Enhanced Data Sources**:
   - Integrate additional data sources such as player statistics, weather conditions, and betting odds to improve model accuracy.
   - Utilize real-time data feeds for live match predictions.

2. **Model Optimization**:
   - Experiment with different machine learning algorithms and neural network architectures to further enhance predictive performance.
   - Implement hyperparameter tuning and automated machine learning (AutoML) techniques.

3. **Feature Engineering**:
   - Develop more sophisticated feature engineering techniques, such as advanced time-series analysis and player-level metrics.
   - Incorporate domain-specific knowledge to create custom features that capture the nuances of football matches.

4. **User Interface**:
   - Create a user-friendly web or mobile application to make predictions accessible to a broader audience.
   - Develop interactive dashboards for visualizing predictions and model insights.

5. **Scalability**:
   - Optimize the system for scalability to handle larger datasets and more frequent updates.
   - Implement distributed computing techniques to speed up data processing and model training.

6. **Model Interpretability**:
   - Enhance model interpretability by incorporating techniques such as SHAP (SHapley Additive exPlanations) values to explain individual predictions.
   - Provide detailed documentation and visualizations to help users understand how the model makes predictions.

7. **Performance Monitoring**:
   - Set up continuous monitoring of model performance to detect and address any degradation over time.
   - Implement automated retraining pipelines to keep the model up-to-date with the latest data.

8. **Community Contributions**:
   - Encourage contributions from the open-source community to add new features, improve existing functionality, and share best practices.
   - Organize hackathons or collaborative events to foster innovation and collaboration.

By pursuing these future directions, the project can continue to evolve and provide even more accurate and valuable predictions for football matches.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit pull requests with improvements.

## 📧 Contact Information

For support or questions, please contact the project maintainers at:

- Email: <szitar.9@gmail.com>


