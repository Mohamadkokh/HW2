FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY ../../feature_engineering/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and helpers
COPY ../../feature_engineering/feature_engineering.py .
COPY ../../feature_engineering/helpers.py .

# Copy the shared common module from the project root
COPY ../../common.py .

CMD ["python", "/app/feature_engineering.py"]
