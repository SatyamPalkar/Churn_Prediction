# File: run_model.py
import pandas as pd
from src.model_training import train_model
from src.evaluation import evaluate_model

# Path to your data file
file_path = 'data/raw/churn_data.csv'

# Train the model
rf_model, X_test_scaled, y_test = train_model(file_path)

# Evaluate the model
evaluate_model(rf_model, X_test_scaled, y_test)
