# Use the official Python image
FROM python:3.9

# SECURITY FIX: Create a standard unprivileged user (UID 1000) required by Hugging Face
RUN useradd -m -u 1000 user
USER user

# Set up the user's home directory path
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Create the working directory inside the new user's home folder
WORKDIR $HOME/app

# Copy the requirements file and grant ownership to the user
COPY --chown=user requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all your project files and grant ownership to the user
COPY --chown=user . .

# Force TensorFlow to use the correct legacy rules to read your custom .keras file
ENV TF_USE_LEGACY_KERAS=1

# Start the server on port 7860
CMD ["uvicorn", "api.py:app", "--host", "0.0.0.0", "--port", "7860"]
