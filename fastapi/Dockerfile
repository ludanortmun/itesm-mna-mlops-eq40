# Base image
FROM python:3.11-slim

# Working directory
WORKDIR /app


# Install dependencies
RUN pip install fastapi uvicorn scikit-learn pydantic

# Copy the proyect
COPY .. /app

ENV PYTHONPATH=/app

# Expose the port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
