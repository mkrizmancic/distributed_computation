import pickle
import socket
import sys

import torch
import yaml
import zmq
from torch_geometric.utils import to_networkx


def send_config_to_node(host, port, config, context: zmq.Context):
    """
    Send a pickled config to a node using a REQ/REP exchange.
    The node binds a REP socket at tcp://*:<port> and replies with a small ACK.
    """
    addr = f"tcp://{host}:{port}"
    sock = context.socket(zmq.REQ)
    # Avoid lingering connections on close so we can quickly talk to many nodes
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.connect(addr)
        sock.send(pickle.dumps(config))
        # Expect an ACK to ensure delivery
        _ = sock.recv()  # b"OK"
    finally:
        sock.close()


def load_model(config):
    model_config = torch.load(config["model"]["path"], weights_only=False)
    return model_config


def load_data(config):
    # Load the dataset and the specific test data index.
    dataset = torch.load(config["test_data"]["path"], weights_only=False)
    test_data = dataset[config["test_data"]["index"]]

    # Extract neighboring nodes for each node in the test data.
    list_of_nodes = list(config["nodes"].keys())
    neighbors = {k: [] for k in config["nodes"]}
    G = to_networkx(test_data, to_undirected=True)

    features = {}

    for node in G.nodes:
        # Populate the neighbors dictionary with the node addresses and ports.
        for neighbor in G.neighbors(node):
            addr_and_port = config["nodes"][list_of_nodes[neighbor]].copy()
            if addr_and_port[0] == "same_as_central":
                addr_and_port[0] = f"{socket.gethostname()}.local"
            neighbors[list_of_nodes[node]].append(addr_and_port)

        # Add the node's initial feature vector.
        features[list_of_nodes[node]] = test_data.x[node]

    return neighbors, features


def main():
    if len(sys.argv) == 2:
        config_file = f"config/{sys.argv[1]}"
    else:
        config_file = "config/config.yaml"
    with open(config_file) as f:
        initial_config = yaml.safe_load(f)

    model_config = load_model(initial_config)
    neighbors, features = load_data(initial_config)

    context = zmq.Context.instance()

    for node_id, addr_port in initial_config["nodes"].items():
        host, port = addr_port[0], addr_port[1]
        if host == "same_as_central":
            host = f"{socket.gethostname()}.local"
        print(f"Sending config to {node_id} at {host}:{port}")
        config = {"model": model_config, "neighbors": neighbors[node_id], "features": features[node_id]}
        try:
            send_config_to_node(host, port, config, context)
        except Exception as e:
            print(f"Failed to send config to {node_id}: {e}")


if __name__ == "__main__":
    main()
