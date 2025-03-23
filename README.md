# ML Pipeline Homework 02

This project implements a complete machine learning pipeline as described in the homework assignment. The pipeline consists of three main steps:

- **Feature Engineering**
- **Training**
- **Inference**

Each step is containerized using Docker, and the entire pipeline is orchestrated with Docker Compose.

> **Note:**  
> Due to exhausted Google Cloud Storage (GCS) credits, this implementation runs entirely locally. All file I/O is handled via a local `data` folder instead of GCS.

---

## Project Structure
.
├── 02_homework_instructions.pdf         # Homework instructions
├── README.md                            # This file
├── common.py                            # Shared module with class definitions (MyPreprocessing, MyClassifier)
├── data                                 # Local folder for input and output data files
│   ├── df_train.csv                     # Training data file
│   ├── df_test_nolabel.csv              # Test data file
│   ├── preprocessor.pkl                 # Pickled preprocessor (generated)
│   ├── X_train.npy                      # Preprocessed training features (generated)
│   ├── y_train.csv                      # Training labels (generated)
│   ├── trained_model_YYYYmmdd_HHMM.pkl   # Timestamped trained model (generated)
│   └── accepted_model.pkl               # Accepted model (generated if F1 score > 0.85)
├── docker                                 # Docker configuration folder
│   ├── feature_engineering
│   │   └── Dockerfile                   # Dockerfile for the feature engineering step
│   ├── training
│   │   └── Dockerfile                   # Dockerfile for the training step
│   └── inference
│       └── Dockerfile                   # Dockerfile for the inference step
├── docker-compose.yml                   # Docker Compose file to run the pipeline
├── feature_engineering                  # Feature engineering code folder
│   ├── feature_engineering.py           # Preprocessing script
│   ├── helpers.py                       # Helper functions (GCS code commented out)
│   └── requirements.txt                 # Python dependencies for feature engineering
├── training                             # Training code folder
│   ├── training.py                      # Training script
│   ├── helpers.py                       # Helper functions (GCS code commented out)
│   └── requirements.txt                 # Python dependencies for training
└── inference                          # Inference code folder
    ├── inference.py                     # Inference script
    ├── helpers.py                       # Helper functions (GCS code commented out)
    └── requirements.txt                 # Python dependencies for inference


---

## Pipeline Details

### 1. Feature Engineering

- **Input:**  
  Reads `df_train.csv` from the local `data` folder.

- **Processing:**  
  Uses the `MyPreprocessing` class (imported from `common.py`) to:
  - Impute missing values
  - Scale numerical features
  - One-hot encode categorical features

- **Output:**  
  Writes the following artifacts to the `data` folder:
  - `preprocessor.pkl`
  - `X_train.npy`
  - `y_train.csv`

### 2. Training

- **Input:**  
  Loads preprocessed artifacts (`X_train.npy` and `y_train.csv`) from the `data` folder.

- **Processing:**  
  - Splits the data into training and validation sets (90%/10%).
  - Trains a Logistic Regression model using `MyClassifier` (from `common.py`).
  - Calculates the F1 score on the validation set.

- **Output:**  
  - Saves a timestamped model file (e.g., `trained_model_YYYYmmdd_HHMM.pkl`) to the `data` folder.
  - If the F1 score is above 0.85, also saves an accepted model as `accepted_model.pkl`.

### 3. Inference

- **Input:**  
  Loads test data (`df_test_nolabel.csv`), along with `preprocessor.pkl` and `accepted_model.pkl` from the `data` folder.

- **Processing:**  
  - Uses the preprocessor to transform the test data.
  - Runs the accepted model to generate predictions.

- **Output:**  
  Saves predictions to `predictions.csv` in the `data` folder.

---

## Docker & Docker Compose Setup

### Dockerfiles

Each component has its own Dockerfile located under the corresponding folder in `docker/`. All Dockerfiles:

- Use the `python:3.9-slim` base image.
- Copy the component code, dependencies, and the shared `common.py` from the project root into the container.
- Specify a command to run the appropriate Python script from the `/app` directory.

### Docker Compose
The docker-compose.yml file orchestrates all three containers. It mounts the local ./data folder to /data inside each container, ensuring that all components share the same data artifacts.

---

## How to use this pipeline locally

### Build and run the pipeline
From the root directory run
''' docker-compose up --build -d '''

Verify execution
''' docker-compose logs'''

Troubleshooting
''' docker-compose logs <container_name> '''