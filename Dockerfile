# Use an official lightweight Python image
FROM python:3.9-slim

# Set up a working directory inside the container
WORKDIR /code

# Copy requirements file first to take advantage of caching
COPY requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all remaining source code into the container
COPY . .

# Hugging Face Spaces always expects traffic routed to port 7860
CMD ["uvicorn", "api.py:app", "--host", "0.0.0.0", "--port", "7860"]