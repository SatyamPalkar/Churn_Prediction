# File: run_hyperparameter_tuning.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from src.data_preprocessing import load_data, preprocess_data, scale_features
from models.hyperparameter_tuning import tune_hyperparameters
from src.evaluation import evaluate_model
from imblearn.over_sampling import SMOTE
import logging

# Set up logging
logging.basicConfig(
    filename='logs/hyperparameter_tuning.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def main():
    # Load and preprocess the data
    file_path = 'data/raw/churn_data.csv'
    df = load_data(file_path)
    df_cleaned = preprocess_data(df)

    # Separate features and target
    X = df_cleaned.drop(columns=['Churn?'])
    y = df_cleaned['Churn?']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Balance the training data using SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Scale the features
    X_train_scaled, X_test_scaled = scale_features(X_train_balanced, X_test)

    # Hyperparameter tuning
    logging.info("Starting hyperparameter tuning...")
    best_model, best_params = tune_hyperparameters(X_train_scaled, y_train_balanced)
    logging.info(f"Best Hyperparameters: {best_params}")

    # Save the best model
    pd.to_pickle(best_model, 'models/saved_models/random_forest_tuned.pkl')

    # Evaluate the model
    logging.info("Evaluating the tuned model...")
    y_pred = best_model.predict(X_test_scaled)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)

    # Log the results
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"AUC-ROC: {auc_roc:.4f}")

    # Print the results
    print("\nBest Hyperparameters:")
    print(best_params)

    print("\nModel Evaluation Metrics (After Hyperparameter Tuning):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    # Save evaluation metrics to a new report
    with open('reports/model_evaluation_tuned.txt', 'w') as f:
        f.write("Model Evaluation Metrics (After Hyperparameter Tuning):\n")
        f.write(f"Best Hyperparameters: {best_params}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUC-ROC: {auc_roc:.4f}\n")

if __name__ == "__main__":
    main()
