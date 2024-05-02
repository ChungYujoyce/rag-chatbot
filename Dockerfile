# Use the chromadb/chroma image as the base image
FROM python:3.11-slim-bookworm AS builder
RUN pip install chromadb

# Expose port 8000

EXPOSE 80

# Run the command to start Chroma on port 8000
CMD ["chroma", "run", "--path", "/chroma_db", "--host", "0.0.0.0", "--port", "8000"]
