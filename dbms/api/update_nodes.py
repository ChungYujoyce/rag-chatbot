import pickle

def update_nodes(_id, new_nodes, DB_PATH, mode):
    with open(f"{DB_PATH}/nodes.pickle", 'rb') as f:
        nodes = pickle.load(f)

    if mode == "add":
        print(len(nodes), len(new_nodes))
        tmp = nodes + new_nodes
        with open(f"{DB_PATH}/nodes.pickle", 'wb') as f:
            pickle.dump(tmp, f)
    else:
        if mode == "edit":
            for node in nodes:
                if node.id_ == _id:
                    node.text = new_nodes['revise_text']
                    node.embedding = new_nodes['revise_vector']
        else:
            nodes = [node for node in nodes if node.id_ != _id]

        with open(f"{DB_PATH}/nodes.pickle", 'wb') as f:
            pickle.dump(nodes, f)