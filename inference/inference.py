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
from common import MyPreprocessing, MyClassifier

# TODO: add helpers import
# from helpers import download_file, upload_file

# TODO: assign bucket name to the BUCKET_NAME variable
BUCKET_NAME = "your-gcs-bucket-name"  # Unused for local execution

if __name__ == "__main__":
    # --- GCS Download (Commented Out) ---
    # download_file(BUCKET_NAME, 'df_test_nolabel.csv', 'df_test_nolabel.csv')
    # download_file(BUCKET_NAME, 'preprocessor.pkl', 'preprocessor.pkl')
    # download_file(BUCKET_NAME, 'accepted_model.pkl', 'accepted_model.pkl')
    
    # --- Local File Load ---
    # Ensure that "df_test_nolabel.csv", "preprocessor.pkl", and "accepted_model.pkl" exist in /data.
    df_test = pd.read_csv("/data/df_test_nolabel.csv")
    
    # Load the preprocessor saved during feature engineering from /data
    preproc = joblib.load("/data/preprocessor.pkl")
    
    # Check if accepted model exists in /data
    if not os.path.exists("/data/accepted_model.pkl"):
        print("Accepted model not found. Please ensure a valid model exists.")
        exit(1)
    model = joblib.load("/data/accepted_model.pkl")
    
    # Preprocess test data
    X_test = preproc.transform(df_test)
    
    # Generate predictions using the accepted model
    predictions = model.predict(X_test)
    pred_df = pd.DataFrame(predictions, columns=["prediction"])
    
    # Save predictions locally to /data
    pred_df.to_csv("/data/predictions.csv", index=False)
    print("Predictions saved to /data/predictions.csv")
    
    # --- GCS Upload (Commented Out) ---
    # upload_file(BUCKET_NAME, 'predictions.csv', 'predictions.csv')