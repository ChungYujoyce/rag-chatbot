import sys, pickle
import argparse
import requests
import os, shutil
import chromadb
import tempfile
from flask_bootstrap import Bootstrap
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from threading import Lock
from api import process_pipeline, update_nodes
from llama_index.embeddings.instructor import InstructorEmbedding
EMBEDDING = InstructorEmbedding(model_name="hkunlp/instructor-xl")
request_lock = Lock()

app = Flask(__name__)
API_HOST = "http://localhost:4105/api"
# SOURCE_DOCUMENTS
source_path = os.path.join(os.path.dirname(__file__), '..', './SOURCE_DOCUMENTS')
DB_PATH = os.path.join(os.path.dirname(__file__), '..', './chroma_db_v1')
DB_PATH_DS = os.path.join(os.path.dirname(__file__), '..', './chroma_db_v1_ds')
chroma_client = chromadb.PersistentClient(DB_PATH)
chroma_client2 = chromadb.PersistentClient(DB_PATH_DS)
DB = chroma_client.get_collection("test_v1")
DB_DS = chroma_client2.get_collection("test_v1")

Bootstrap(app)

def get_processed_items():
    data = DB.get(include=['documents', 'metadatas', 'embeddings'])
    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]

    combined_data = []
    for id_, meta, doc in zip(ids, metadatas, documents):
        combined_data.append({
            "id": id_,
            "source": meta["file_name"],
            "document": doc
        })
    return combined_data


@app.route('/')
def index():
    items = get_processed_items()
    return render_template('dataList.html', items=items)
 

def update_page():
    items = get_processed_items()
    return render_template(
        "dataList.html",
        show_response_modal=False,
        response_dict={"Prompt": "None", "Answer": "None", "Sources": [("ewf", "wef")]},
        items=items
    )


@app.route("/", methods=["GET", "POST", "PUT", "DELETE"])
def main_page():
    app.logger.info(request)
    if request.method == "POST":
        if request.form.get("action") == "add" and "documents" in request.files:
            upload_url = f"{API_HOST}/upload"
            files = request.files.getlist("documents")
            for file in files:
                print(file.filename)
                filename = secure_filename(file.filename)
                with tempfile.SpooledTemporaryFile() as f:
                    f.write(file.read())
                    f.seek(0)
                    response = requests.post(upload_url, files={"document": (filename, f)})
            
    elif request.method == "PUT":
        if "editInput" in request.form:
            _id, revise_result = request.form.get("id"), request.form.get("revise_result")
            revise_result_url = f"{API_HOST}/run_update"
            response = requests.put(revise_result_url, data={"id": _id, "revise_result": revise_result})
        
    elif request.method == "DELETE":
        if "deleteInput" in request.form:
            _id = request.form.get("id")
            delete_url = f"{API_HOST}/run_delete"
            response = requests.delete(delete_url, data={"id": _id})

    print(response.text)
    
    return update_page()


@app.route("/api/upload", methods=["POST"])
def save_document_route():
    try:
        global request_lock
        if "document" not in request.files:
            return "No document part", 400
        file = request.files["document"]
        if file.filename == "":
            return "No selected file", 400
        if file:
            filename = secure_filename(file.filename)
            
            # check if the file already exists.
            with request_lock:
                results = DB.get(where={"file_name": f"{filename}"})
            if len(results["ids"]) > 0:
                print(results["ids"])
                print("Document already exists. Skipping...")
                return "Document already exists. Skipping...", 200
            else:
                print("Loading document...")

                # create a tmp folder to save the file
                folder_path = './uploads'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                file_path = os.path.join(folder_path, filename)
                file.save(file_path)
                # Perform data processing on the file
                text_nodes = process_pipeline(folder_path, DB_PATH)

                # update main pickle
                update_nodes("", text_nodes, DB_PATH, "add")

                # update datasheet pickle additionally if it's a product datasheet
                product_prefix = ['ars-', 'as-', 'asg-', 'ssg-', 'sys-']
                if any(prefix in filename for prefix in product_prefix):
                    print("add ds")
                    update_nodes("", text_nodes, DB_PATH_DS, "add")
                    for text_node in text_nodes:
                        DB_DS.add(ids=text_node.id_,
                                metadatas=text_node.metadata,
                                documents=text_node.text,
                                embeddings=text_node.embedding)

                for text_node in text_nodes:
                    DB.add(ids=text_node.id_,
                            metadatas=text_node.metadata,
                            documents=text_node.text,
                            embeddings=text_node.embedding)
                
                # move the uploaded file to SOURCE_DOCUMENT
                shutil.copy(file_path, source_path)
                os.remove(file_path)
            
                # Return the processed data as JSON response
                return"Upload file successfully", 200
    except Exception as e:
        app.logger.info(str(e))
        return f"Error occurred: {str(e)}", 500


@app.route("/api/run_update", methods=["PUT"])
def run_update():
    try:
        global request_lock  # Make sure to use the global lock instance
        _id, revise_result = request.form.get("id"), request.form.get("revise_result")
        app.logger.info(_id)
        app.logger.info(revise_result)

        revise_text = revise_result.strip()
        revise_vector = EMBEDDING._get_text_embedding(revise_text)

        with request_lock:
            DB.update(ids=_id, embeddings=revise_vector, documents=revise_text)
            if DB_DS.get(ids=_id):
                DB_DS.update(ids=_id, embeddings=revise_vector, documents=revise_text)

        data = DB.get(ids=_id)
        filename = data['metadatas'][0]['file_name']

        # update pickle nodes
        new_nodes = {
            'revise_text': f"## {filename.split('.')[0]}\n{revise_text}",
            'revise_vector': revise_vector
        }
        update_nodes(_id, new_nodes, DB_PATH, "edit")
        # update ds pickle nodes
        product_prefix = ['ars-', 'as-', 'asg-', 'ssg-', 'sys-']
        if any(prefix in filename for prefix in product_prefix):
            print("update ds")
            update_nodes(_id, new_nodes, DB_PATH_DS, "edit")

        return "Script executed successfully", 200
    except Exception as e:
        app.logger.info(str(e))
        return f"Error occurred: {str(e)}", 500


@app.route("/api/run_delete", methods=["DELETE"])
def run_delete():
    try:
        global request_lock  # Make sure to use the global lock instance
        _id = request.form.get("id")
        app.logger.info(_id)

        data = DB.get(ids=_id)
        product_prefix = ['ars-', 'as-', 'asg-', 'ssg-', 'sys-']
        filename = data['metadatas'][0]['file_name']
        if any(prefix in filename for prefix in product_prefix):
            print("delete ds")
            update_nodes(_id, {}, DB_PATH_DS, "delete")

        update_nodes(_id, {}, DB_PATH, "delete")

        with request_lock:
            print(f"delete id:{_id}")
            DB.delete(ids=_id)
            if DB_DS.get(ids=_id):
                DB_DS.delete(ids=_id)

        return "Script executed successfully", 200
    except Exception as e:
        return f"Error occurred: {str(e)}", 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4105, help="Port to run the UI on. Defaults to 5000.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the UI on. Defaults to 127.0.0.1. "
        "Set to 0.0.0.0 to make the UI externally "
        "accessible from other devices.",
    )
    args = parser.parse_args()
    app.run(debug=True, host=args.host, port=args.port)
