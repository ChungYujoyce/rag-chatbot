Integrate Llamaindex, ChromaDB, Chainlit to build an advanced chatbot (an upgrade version of my MicroGPT repo)

### Sample Qeustion you can ask:

- [FAQ](https://docs.google.com/spreadsheets/d/12sph9aLuE-3MHstkoZlYy3Rvz0Jq7mBzkzVr7VhH9QQ/edit#gid=522552108)

- [General Questions](https://docs.google.com/document/d/1kyhXFBtrDSIWkY6XjuqK7DXZ0cUEBqiZxg0kCyNDvso/edit?usp=sharing)

#### Please note that if changing a topic, refresh the page or click 'New Chat' button on the top right.

### Get the Repo
```
git clone https://github.com/ChungYujoyce/rag-chatbot.git
```

### Go to [link](https://mysupermicro-my.sharepoint.com/personal/joyce_huang_supermicro_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjoyce%5Fhuang%5Fsupermicro%5Fcom%2FDocuments%2FSuperGPT&view=0) and download the folders, place them in the repo.
- chroma_db_v1
- chroma_db_v1_ds
- csvtest
- SOURCE_DOCUMENTS


### Llama3 8B vLLM Server

```
docker pull docker.io/jj0122/microgpt:vllmv0.4.1
```
```
docker run -it --network my_network --name vllm jj0122/microgpt:vllmv0.4.1  /bin/bash
```
### In the docker, run below commands to start the server
```
cd rag-chatbot/server
bash 1_start_server_vllm.sh
```

### ChatBot Client
```
docker pull docker.io/jj0122/rag_app:0522
```
```
docker run -it --network my_network --name index jj0122/rag_app:0522   /bin/bash
```
### In the docker, run below commands to start the client (the port number in the bash file should be changed to an appplicable one)
```
cd rag-chatbot/
bash run-test.sh
```

### Stop the Client
```
bash clean.sh
```