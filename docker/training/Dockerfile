FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY ../../training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training code and helpers
COPY ../../training/training.py .
COPY ../../training/helpers.py .

# Copy the shared module from the project root
COPY ../../common.py .

# Run the training script
CMD ["python", "/app/training.py"]
