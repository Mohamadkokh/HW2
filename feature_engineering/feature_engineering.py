from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np
import joblib
import os
from common import MyPreprocessing

# TODO: add helpers import
# from helpers import download_file, upload_file

# TODO: assign bucket name to the BUCKET_NAME variable
BUCKET_NAME = "your-gcs-bucket-name"  # Not used for local execution

if __name__ == "__main__":
    # --- GCS Download (Commented Out) ---
    # download_file(BUCKET_NAME, 'df_train.csv', 'df_train.csv')
    
    # --- Local File Read ---
    # Ensure "df_train.csv" is in /data.
    X_train = pd.read_csv("/data/df_train.csv")
    
    # Define the target column
    target_col = "Disease"
    y_train = X_train[[target_col]]
    X_train = X_train[[col for col in X_train.columns if col != target_col]]
    
    # Preprocess the training data using MyPreprocessing from common.py
    preproc = MyPreprocessing()
    preproc.fit(X_train)
    X_train_preproc = preproc.transform(X_train)
    
    # Save the preprocessor, processed features, and target to /data
    joblib.dump(preproc, "/data/preprocessor.pkl")
    np.save("/data/X_train.npy", X_train_preproc)
    y_train.to_csv("/data/y_train.csv", index=False)
    
    # --- GCS Upload (Commented Out) ---
    # upload_file(BUCKET_NAME, 'y_train.csv', 'y_train.csv')
    # upload_file(BUCKET_NAME, 'X_train.npy', 'X_train.npy')
    # upload_file(BUCKET_NAME, 'preprocessor.pkl', 'preprocessor.pkl')
    
    print("Feature engineering completed and files saved to /data.")

