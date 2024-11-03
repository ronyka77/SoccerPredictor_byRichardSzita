# Football Predictor by Richard Szita

This project is a comprehensive machine learning pipeline designed to predict football match outcomes. It leverages a combination of ensemble learning techniques and neural networks to enhance predictive accuracy.

## Features

- Data Preprocessing: Handles missing values using iterative imputation and scales features for optimal model performance.
- Feature Selection: Employs Recursive Feature Elimination (RFE) to identify the most significant features.
- Model Training: Utilizes a stacking regressor that combines Random Forest, Support Vector Regressor (SVR), Neural Network, and XGBoost models.
- Custom Metrics: Implements a custom metric to evaluate predictions within a specified tolerance range.
- Model Persistence: Saves trained models using `joblib` and `cloudpickle` for future use.

## Installation

1. Clone the Repository:

   bash
   git clone <https://github.com/ronyka77/Football_predictor_byRichardSzita.git>

2. Navigate to the Project Directory:

   bash
   cd Football_predictor_byRichardSzita

3. Create and Activate a Virtual Environment:

   bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
4. Install Dependencies:

   bash
   pip install -r requirements.txt

## Usage

1. Data Preparation: Ensure your dataset is in the correct format as expected by the script.

2. Model Training: Run the training script to preprocess data, select features, and train the stacking regressor.

   bash
   python train_model.py

   

3. Model Evaluation: Evaluate the model's performance using metrics such as Mean Squared Error (MSE), R-squared, Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and the custom within-range metric.

4. Prediction: Use the trained model to make predictions on new data.

   bash
   python predict.py

## Model Saving and Loading

The project saves the trained models in two parts:

- Neural Network Model: Saved separately in H5 format.
- Stacking Regressor: Saved using `cloudpickle` after removing the neural network component.

This approach ensures that custom objects and layers are appropriately handled during the saving and loading process.

## Custom Metrics

The project includes a custom metric, `within_range_metric`, which evaluates the percentage of predictions within a specified tolerance range. This metric is integrated into the Keras model and is essential for assessing model performance.

## Logging

The training and prediction processes are logged extensively, providing insights into data preprocessing steps, model training progress, and evaluation metrics. Logs are saved in the `./Model_build1/score_prediction/log/` directory.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License.

## Acknowledgments

Special thanks to the open-source community for providing the tools and libraries that made this project possible.
