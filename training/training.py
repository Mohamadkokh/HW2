from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import joblib
import datetime
import os
from common import MyClassifier

# TODO: add helpers import
# from helpers import download_file, upload_file

# TODO: assign bucket name to the BUCKET_NAME variable
BUCKET_NAME = "your-gcs-bucket-name"  # Unused for local execution

if __name__ == "__main__":
    # --- GCS Download (Commented Out) ---
    # download_file(BUCKET_NAME, 'X_train.npy', 'X_train.npy')
    # download_file(BUCKET_NAME, 'y_train.csv', 'y_train.csv')
    
    # --- Local File Load ---
    X = np.load("/data/X_train.npy")
    y = pd.read_csv("/data/y_train.csv")
    
    # Split data into training and validation sets (90%/10%)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Train classifier on the training set using MyClassifier from common.py
    clf = MyClassifier()
    clf.fit(X_train, y_train.values.ravel())
    
    # Validate the model on the validation set
    y_pred = clf.predict(X_val)
    score = f1_score(y_val, y_pred)
    print(f"Validation F1 Score: {score}")
    
    # Save the trained model with a timestamp in the filename to /data
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_filename = f"/data/trained_model_{timestamp}.pkl"
    joblib.dump(clf, model_filename)
    print(f"Model saved as {model_filename}")
    
    # --- GCS Upload (Commented Out) ---
    # upload_file(BUCKET_NAME, model_filename, model_filename)
    
    # If F1 score exceeds threshold, save an accepted model version to /data
    if score > 0.85:
        accepted_model_filename = "/data/accepted_model.pkl"
        joblib.dump(clf, accepted_model_filename)
        print("Accepted model saved as /data/accepted_model.pkl")
        # --- GCS Upload (Commented Out) ---
        # upload_file(BUCKET_NAME, accepted_model_filename, accepted_model_filename)