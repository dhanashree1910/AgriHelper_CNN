FROM python:3.10-slim

# Create and set working directory
WORKDIR /app

# Copy everything from your project into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose default FastAPI port
EXPOSE 7860

# Command to start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
