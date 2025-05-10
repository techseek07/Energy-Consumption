🔌 PJME Hourly Energy Demand Forecasting
This project performs time series forecasting on hourly energy demand data from the PJME (PJM East) region using machine learning models such as XGBoost, Random Forest, and an ensemble approach. It utilizes time-based and statistical features to predict future energy consumption, helping stakeholders with load forecasting and resource planning.

📂 Dataset
Source: PJME Hourly Energy Data

File: PJME_hourly.csv

Features:

Datetime: Timestamp (hourly)

PJME_MW: Megawatts (MW) of electricity used

📈 Models Used
XGBoost Regressor

Random Forest Regressor

Ensemble (Average of XGBoost & RF predictions)

🧠 Feature Engineering
Includes:

Time-based features: Hour, day of week, month, etc.

Lag features: Previous hour and previous day

Rolling statistics: Mean and standard deviation over a 6-hour window

✅ Evaluation Metrics
Root Mean Squared Error (RMSE)

Mean Absolute Error (MAE)

🛠️ Installation & Requirements
Run this notebook in Google Colab. The necessary packages are pre-installed in most environments:

bash
Copy
Edit
pip install xgboost seaborn scikit-learn matplotlib pandas
🚀 How to Use
1. Upload the dataset:
Ensure the file PJME_hourly.csv is available in your Colab environment (e.g., /content/).

2. Run the Notebook:
The script will:

Visualize the dataset

Engineer features

Train and evaluate both XGBoost and Random Forest models

Provide comparative performance metrics

Plot predictions vs actual values

3. Example Output:
yaml
Copy
Edit
Model Performance Results:

RANDOM_FOREST Model:
RMSE: 1150.45
MAE : 800.12

XGBOOST Model:
RMSE: 1120.33
MAE : 785.67

ENSEMBLE Model:
RMSE: 1087.20
MAE : 770.45
📊 Visualizations
Hourly & monthly distribution of energy usage

Training vs test data split

Actual vs predicted plots (zoomed in & full scale)

Feature importance plots

🧪 File Structure
plaintext
Copy
Edit
├── PJME_hourly.csv           # Hourly energy dataset
├── EnergyPredictionModel     # Class-based implementation (Random Forest & XGBoost)
├── Visualization Section     # Raw vs Predicted
├── Ensemble Evaluation       # Combined predictions
🤝 Contributing
If you'd like to contribute improvements, optimizations, or alternative model integrations, feel free to fork and open a pull request.

