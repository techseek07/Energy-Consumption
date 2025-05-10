import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
df = pd.read_csv('/content/PJME_hourly.csv')
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)

df.plot(style='.',
        figsize=(15, 5),
        color=color_pal[0],
        title='PJME Energy Use in MW')
plt.show()


train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline('01-01-2015', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(df)

fig, ax = plt.subplots(figsize=(7, 7))
sns.boxplot(data=df, x='hour', y='PJME_MW')
ax.set_title('MW by Hour')
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='month', y='PJME_MW')
ax.set_title('MW by Month')
plt.show()

train = create_features(train)
test = create_features(test)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'PJME_MW'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:squarederror',  # updated objective
                       max_depth=3,
                       learning_rate=0.01)

reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)

fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
plt.show()

test['prediction'] = reg.predict(X_test)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
ax = df[['PJME_MW']].plot(figsize=(10, 5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Data and Prediction')
plt.show()

ax = df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['PJME_MW'] \
    .plot(figsize=(10, 5), title='Week Of Data')
df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['prediction'] \
    .plot(style='.')
plt.legend(['Truth Data','Prediction'])
plt.show()

score = np.sqrt(mean_squared_error(test['PJME_MW'], test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'PJME_MW'

X_train = train[FEATURES]
y_train = train[TARGET]
X_test = test[FEATURES]
y_test = test[TARGET]

# Initialize and train Random Forest model
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
rf.fit(X_train, y_train)

# Predict and calculate RMSE for Random Forest
test['rf_prediction'] = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, test['rf_prediction']))
print(f'Random Forest RMSE Score on Test set: {rf_rmse:0.2f}')

# Visualize Random Forest predictions
ax = df[['PJME_MW']].plot(figsize=(10, 5), title='Raw Data and Predictions')
df['prediction'].plot(ax=ax, style='.', color='blue', label='XGBoost Prediction')
test['rf_prediction'].plot(ax=ax, style='.', color='green', label='Random Forest Prediction')
plt.legend(['Truth Data', 'XGBoost Prediction', 'Random Forest Prediction'])
plt.show()

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class EnergyPredictionModel:
    def _init_(self):  # Correct constructor
        # Initialize models
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()

    def create_features(self, df):
        """
        Create comprehensive time-based features

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with additional features
        """
        df = df.copy()

        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear

        # Lag features
        df['lag_1'] = df['PJME_MW'].shift(1)
        df['lag_24'] = df['PJME_MW'].shift(24)

        # Rolling features
        df['rolling_mean_6h'] = df['PJME_MW'].rolling(window=6, min_periods=1).mean()
        df['rolling_std_6h'] = df['PJME_MW'].rolling(window=6, min_periods=1).std()

        return df.dropna()

    def prepare_data(self, df, features):
        """
        Prepare features and target for modeling

        Args:
            df (pd.DataFrame): Input dataframe
            features (list): List of feature columns

        Returns:
            Tuple of scaled features and target
        """
        X = df[features]
        y = df['PJME_MW']

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def train_and_evaluate(self, df):
        """
        Train models and evaluate their performance

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            Dict of model performances
        """
        # Create features
        df_features = self.create_features(df)

        # Select features
        features = [
            'hour', 'day_of_week', 'month', 'day_of_year',
            'lag_1', 'lag_24', 'rolling_mean_6h', 'rolling_std_6h'
        ]

        # Prepare data
        X, y = self.prepare_data(df_features, features)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Train Random Forest
        self.rf_model.fit(X_train, y_train)
        rf_pred = self.rf_model.predict(X_test)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_mae = mean_absolute_error(y_test, rf_pred)

        # Train XGBoost
        self.xgb_model.fit(X_train, y_train)
        xgb_pred = self.xgb_model.predict(X_test)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_mae = mean_absolute_error(y_test, xgb_pred)

        # Ensemble Prediction
        ensemble_pred = 0.5 * rf_pred + 0.5 * xgb_pred
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)

        return {
            'random_forest': {
                'rmse': rf_rmse,
                'mae': rf_mae
            },
            'xgboost': {
                'rmse': xgb_rmse,
                'mae': xgb_mae
            },
            'ensemble': {
                'rmse': ensemble_rmse,
                'mae': ensemble_mae
            }
        }

def main():
    # Load data
    try:
        df = pd.read_csv('/content/PJME_hourly.csv', parse_dates=['Datetime'], index_col='Datetime')
    except FileNotFoundError:
        print("unable to read")
        return

    # Initialize and run model
    model = EnergyPredictionModel()
    results = model.train_and_evaluate(df)

    # Print results
    print("\nModel Performance Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()} Model:")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAE: {metrics['mae']:.2f}")

if _name_ == "_main_":
    main()


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

class EnergyPredictionModel:
    def _init_(self):
        self.rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42)
        self.scaler = StandardScaler()

    def create_features(self, df):
        df = df.copy()
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        df['lag_1'] = df['PJME_MW'].shift(1)
        df['lag_24'] = df['PJME_MW'].shift(24)
        df['rolling_mean_6h'] = df['PJME_MW'].rolling(window=6, min_periods=1).mean()
        df['rolling_std_6h'] = df['PJME_MW'].rolling(window=6, min_periods=1).std()
        return df.dropna()

    def prepare_data(self, df, features):
        X = df[features]
        y = df['PJME_MW']
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def train_and_visualize(self, df):
        # Create features
        df_features = self.create_features(df)

        features = [
            'hour', 'day_of_week', 'month', 'day_of_year',
            'lag_1', 'lag_24', 'rolling_mean_6h', 'rolling_std_6h'
        ]

        # Prepare data
        X, y = self.prepare_data(df_features, features)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Train models
        self.rf_model.fit(X_train, y_train)
        self.xgb_model.fit(X_train, y_train)

        # Predictions
        rf_pred = self.rf_model.predict(X_test)
        xgb_pred = self.xgb_model.predict(X_test)
        ensemble_pred = 0.5 * rf_pred + 0.5 * xgb_pred

        # Visualization
        plt.figure(figsize=(15, 6))
        plt.plot(y_test.values[:200], label='Actual', color='black', linewidth=2)
        plt.plot(ensemble_pred[:200], label='Ensemble', color='blue', alpha=0.7)

        plt.title('Energy Consumption: Actual vs Predicted')
        plt.xlabel('Time Steps')
        plt.ylabel('Energy Consumption (MW)')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Performance metrics
        print("Model Performance:")
        print(f"Random Forest RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.2f}")
        print(f"XGBoost RMSE: {np.sqrt(mean_squared_error(y_test, xgb_pred)):.2f}")
        print(f"Ensemble RMSE: {np.sqrt(mean_squared_error(y_test, ensemble_pred)):.2f}")

def main():
    # Load data
    df = pd.read_csv('/content/PJME_hourly.csv', parse_dates=['Datetime'], index_col='Datetime')

    # Initialize and run model
    model = EnergyPredictionModel()
    model.train_and_visualize(df)

if _name_ == "_main_":
    main()
