FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY ../../inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference code and helpers
COPY ../../inference/inference.py .
COPY ../../inference/helpers.py .

# Copy the shared module from the project root
COPY ../../common.py .

# Run the inference script
CMD ["python", "/app/inference.py"]
