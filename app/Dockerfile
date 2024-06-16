# Use the official Python image as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Expose the port your app runs on
EXPOSE 8501

# Command to run your application
CMD ["streamlit", "run", "app_redis.py", "--server.port=8501", "--server.address=0.0.0.0"]