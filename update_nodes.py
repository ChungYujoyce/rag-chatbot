import pickle
from constants import DB_PATH

def update_nodes(_id, new_nodes, mode):
    with open(f"{DB_PATH}/nodes.pickle", 'rb') as f:
        nodes = pickle.load(f)

    if mode == "add":
        nodes += new_nodes
    elif mode == "edit":
        for node in nodes:
            if node.node_id == _id:
                node.text = new_nodes['revise_text']
                node.embedding = new_nodes['revise_vector']
    else:
        nodes = [node for node in nodes if node.id_ != _id]

    with open(f"{DB_PATH}/nodes.pickle", 'wb') as f:
        pickle.dump(nodes, f)

print(DB_PATH)


