# Base image
FROM python:3.10-slim

# Working directory
WORKDIR /app

# Copy the proyect
COPY .. /app

# Install our library code

RUN pip install -r requirements.txt
RUN pip install mlops

ENV PYTHONPATH=/app

# Expose the port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "prediction_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
