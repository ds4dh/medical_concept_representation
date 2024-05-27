# Use the official Python image from the Docker Hub
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Install the specified pip packages
RUN pip install --no-cache-dir numpy==1.24.4 pandas==1.5.3 tqdm==4.66.1

# Set the entry point to run the script
ENTRYPOINT ["python", "/app/data/datasets/mimic-iv-2.2/process_mimic.py"]
