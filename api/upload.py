from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict
import os, sys, signal
from ingest import process_pipeline
import logging
import uvicorn
import asyncio


app = FastAPI()
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Set the log message format
    filename="app.log",  # Set the log file
)

# Configuration
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created '{UPLOAD_FOLDER}' folder.")
else:
    print(f"'{UPLOAD_FOLDER}' folder already exists.")

ALLOWED_EXTENSIONS = {'pdf', 'doc', 'txt'}

# Function to handle termination signal and release the port
def cleanup(signal, frame):
    print("\nStopping server...")
    # Add code here to release the port or perform any other cleanup actions
    sys.exit(0)

# Register signal handler for termination signal (SIGTERM or SIGINT)
signal.signal(signal.SIGTERM, cleanup)
signal.signal(signal.SIGINT, cleanup)

@app.get("/")
def read_root():
    logging.debug("Processing request to root endpoint")
    return {"Hello": "World"}

# Function to check if the file extension is allowed
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API endpoint to upload a file and process it
@app.post('/upload')
def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    # If user does not select file, raise an HTTPException
    if not file:
        raise HTTPException(status_code=400, detail='No file provided')

    # Check if the file has an allowed extension
    if allowed_file(file.filename):
        # Save the file to the specified folder
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(filepath, 'wb') as f:
            f.write(file.file.read())

        # Perform data processing on the file
        text_nodes = process_pipeline(UPLOAD_FOLDER)
        json_data = []
        
        for text_node in text_nodes:
            json_data.append({
                'id_': text_node.id_,
                'metadata': text_node.metadata,
                'embedding': text_node.embedding,
                'text': text_node.text,
            })
        # Return the processed data as JSON response
        return JSONResponse(content={'result': json_data}, status_code=200)

    else:
        raise HTTPException(status_code=400, detail='Invalid file type')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=4107)

# async def run_server():
#     uvicorn.run(app, host='0.0.0.0', port=4107)


# async def main():
#     # Run the async function in a separate task
#     task = asyncio.create_task(async_function())

#     # Run the server in another task
#     server_task = asyncio.create_task(run_server())

#     # Wait for both tasks to complete
#     await asyncio.gather(task, server_task)

# # Run the main function using asyncio.run()
# if __name__ == '__main__':
#     asyncio.run(main())