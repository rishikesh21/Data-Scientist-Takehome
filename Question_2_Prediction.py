#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install catboost scikit-learn')


# In[2]:


import requests
import time
import pandas as pd
import numpy as np
from io import StringIO
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.optimize import minimize
import warnings
warnings.filterwarnings(action="ignore")

# Dataset IDs
BIDDING_DATASET_ID = "d_69b3380ad7e51aff3a7dcc84eba52b8a"

def load_dataset_as_dataframe(DATASET_ID):
    """Load dataset from data.gov.sg API"""
    INITIATE_URL = f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/initiate-download"
    POLL_URL = f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/poll-download"

    init_resp = requests.get(INITIATE_URL)
    init_resp.raise_for_status()

    for _ in range(3):
        time.sleep(2)
        poll_resp = requests.get(POLL_URL)
        poll_resp.raise_for_status()
        download_url = poll_resp.json().get("data", {}).get("url")
        if download_url:
            break

    csv_resp = requests.get(download_url)
    csv_resp.raise_for_status()
    return pd.read_csv(StringIO(csv_resp.text))

print("Loading COE data...")
df = load_dataset_as_dataframe(BIDDING_DATASET_ID)


# In[3]:


for col in ["quota", "bids_success", "bids_received", "premium"]:
    df[col] = df[col].astype(str).str.replace(',', '').astype(float)

df['month'] = pd.to_datetime(df['month'])
df = df.sort_values(['vehicle_class', 'month'])


df['demand_supply_ratio'] = df['bids_received'] / df['quota']
df['excess_demand'] = (df['bids_received'] - df['quota']).clip(lower=0)
df['success_rate'] = df['bids_success'] / df['bids_received']

for lag in [1, 2, 3, 6]:
    df[f'premium_lag_{lag}'] = df.groupby('vehicle_class')['premium'].shift(lag)
    df[f'demand_ratio_lag_{lag}'] = df.groupby('vehicle_class')['demand_supply_ratio'].shift(lag)

df['premium_ma_3'] = df.groupby('vehicle_class')['premium'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)
df['demand_ma_3'] = df.groupby('vehicle_class')['demand_supply_ratio'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)

df['year'] = df['month'].dt.year
df['month_num'] = df['month'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

df = df.dropna()


df_model = pd.get_dummies(df, columns=['vehicle_class'], prefix='cat')

feature_cols = ['quota', 'demand_supply_ratio', 'excess_demand', 'success_rate',
                'premium_lag_1', 'premium_lag_2', 'premium_lag_3', 'premium_lag_6',
                'demand_ratio_lag_1', 'demand_ratio_lag_2', 'demand_ratio_lag_3',
                'premium_ma_3', 'demand_ma_3', 'year', 'month_sin', 'month_cos'] + \
               [col for col in df_model.columns if col.startswith('cat_')]

X = df_model[feature_cols]
y = df_model['premium']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=False
)

print("\nTraining XGBoost model...")
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"Model Performance:")
print(f"  RMSE: ${rmse:.2f}")
print(f"  R²: {r2:.4f}")
print(f"  MAPE: {mape:.2f}%")



# In[4]:


latest = df.groupby('vehicle_class').last()
categories = latest.index.values
current_quotas = latest['quota'].values
current_premiums = latest['premium'].values

target = current_premiums * 0.9

def predict_new_premium(new_quota, category_idx):
    X = df_model[df_model[f'cat_{categories[category_idx]}'] == 1].iloc[-1][feature_cols].values
    X[feature_cols.index('quota')] = new_quota
    return xgb_model.predict(scaler.transform(X.reshape(1, -1)))[0]

from scipy.optimize import minimize

def cost(multipliers):
    return sum((predict_new_premium(current_quotas[i] * m, i) - target[i])**2
               for i, m in enumerate(multipliers))

result = minimize(cost, x0=[1.1]*len(categories), bounds=[(0.9, 1.2)]*len(categories))

print("\nOPTIMIZATION RESULTS")
print("-" * 40)
for i, cat in enumerate(categories):
    new_quota = current_quotas[i] * result.x[i]
    new_premium = predict_new_premium(new_quota, i)
    print(f"{cat}: +{(result.x[i]-1)*100:.0f}% quota → ${new_premium:,.0f} ({(new_premium/current_premiums[i]-1)*100:+.0f}%)")

print(f"\nAverage quota increase: {(result.x.mean()-1)*100:.0f}%")


# In[4]:




