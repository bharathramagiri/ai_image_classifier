# Use the official Python image
FROM python:3.9

# SECURITY FIX: Create a standard unprivileged user
RUN useradd -m -u 1000 user
USER user

# Set up the user's home directory path
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Create the working directory
WORKDIR $HOME/app

# Copy the requirements file and grant ownership
COPY --chown=user requirements.txt .

# Install dependencies (This will now grab tensorflow-cpu and tf-keras!)
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all your project files
COPY --chown=user . .

# Force TensorFlow to use the legacy tf-keras package we just installed
ENV TF_USE_LEGACY_KERAS=1

# Start the server on port 7860
CMD ["uvicorn", "api.py:app", "--host", "0.0.0.0", "--port", "7860"]
