import importlib
import pickle
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import zmq
from torch import Tensor

from utils.gnn_splitter import GNNSplitter
from utils.led_matrix import LEDMatrix


class NodeZMQ:
    """
    ZMQ-based node.

    Protocol:
    - Config phase: Node binds REP on tcp://*:<port> and waits for a single config message
      from the central process (REQ). Replies with b"OK" and closes.
    - Runtime phase: Node binds PULL on the same port to receive neighbor messages.
      For sending, it creates one PUSH socket per neighbor and connects to their PULL endpoint.
    """

    def __init__(self, node_id: str):
        self.node_id = f"node{node_id}"  # Just for printing.
        self.port = 5000 + int(node_id)  # Unique per node; we reuse for ZMQ endpoints.

        self.config = {}  # Holds model properties, neighbors, and initial features.

        self.value = Tensor()  # Current node representation.
        self.output = Tensor()  # Interpretable output after each layer.
        self.layer = 0  # Current GNN layer.
        self.received = defaultdict(dict)  # Per-layer values received from neighbors.

        # LED display (works on non-RPi via no-op shim in LEDMatrix implementation)
        self.led = LEDMatrix()

        # ZMQ context and sockets
        self.context = zmq.Context.instance()
        self.pull_sock = None  # PULL bound on our port for runtime messages
        self.push_socks: Dict[Tuple[str, int], zmq.Socket] = {}  # One PUSH per neighbor endpoint

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Always clean up the LED matrix
        self.led.off()

        # Close ZMQ sockets cleanly
        try:
            if self.pull_sock is not None:
                self.pull_sock.setsockopt(zmq.LINGER, 0)
                self.pull_sock.close(0)
            for s in self.push_socks.values():
                s.setsockopt(zmq.LINGER, 0)
                s.close(0)
        finally:
            # Do not terminate global context (it may be shared), but flush I/O
            pass

        # If an exception occurred, re-raise it after cleanup
        if exc_type is not None:
            raise exc_val

    def initialize_GNN(self):
        # Load model configuration from the received config and set up the GNN model.
        model_config = self.config["model"]
        GNN = getattr(importlib.import_module("torch_geometric.nn"), model_config["architecture"])
        self.model = GNN(
            in_channels=model_config["in_channels"],
            hidden_channels=model_config["hidden_channels"],
            num_layers=model_config["num_layers"],
            out_channels=model_config["out_channels"],
        )
        self.model.load_state_dict(model_config["state_dict"])  # type: ignore[arg-type]
        self.distributed_model = GNNSplitter(self.model)

        # Load the initial feature vector for this node.
        self.value = self.config["features"].unsqueeze(0)  # Ensure it's a batch of size 1.
        print(f"Initial feature vector for node {self.node_id}: {self.value}")

    def wait_for_config(self, port: int):
        # Bind a REP socket to receive a single config frame.
        rep = self.context.socket(zmq.REP)
        rep.setsockopt(zmq.LINGER, 0)
        rep.bind(f"tcp://*:{port}")
        print(f"Node {self.node_id} waiting for config on tcp://*:{port}")
        try:
            data = rep.recv()  # blocking
            config = pickle.loads(data)
            rep.send(b"OK")  # ACK
            return config
        finally:
            # Close and free the port so we can bind the runtime socket next
            rep.close(0)

    def setup_runtime_sockets(self):
        # Bind PULL on our port to receive neighbor messages.
        self.pull_sock = self.context.socket(zmq.PULL)
        self.pull_sock.setsockopt(zmq.LINGER, 0)
        self.pull_sock.bind(f"tcp://*:{self.port}")
        print(f"Node {self.node_id} listening for values on tcp://*:{self.port}")

        # Create a dedicated PUSH socket per neighbor and connect to their PULL endpoint.
        for neighbor_addr, neighbor_port in self.config["neighbors"]:
            addr = f"tcp://{neighbor_addr}:{neighbor_port}"
            s = self.context.socket(zmq.PUSH)
            s.setsockopt(zmq.LINGER, 0)
            s.connect(addr)
            self.push_socks[(neighbor_addr, neighbor_port)] = s

        # Slight delay to ensure connections are established across peers
        time.sleep(1.0)

    def start(self):
        # 1) Wait for config from central node (REQ/REP)
        config = self.wait_for_config(self.port)
        self.config = config

        # 2) Set up the GNN model
        self.initialize_GNN()

        # 3) Prepare runtime sockets (PULL inbound, PUSH per-neighbor outbound)
        self.setup_runtime_sockets()

        # 4) Run the layer-by-layer exchange and update
        while self.layer < self.model.num_layers:
            self.exchange_and_update()
            print(f"Layer {self.layer} output: {self.output.item()}\n")

            vmin = 0
            vmax = 4
            norm_output = max(0.0, min(1.0, (self.output.item() - vmin) / (vmax - vmin)))

            self.led.set_percentage(norm_output, 800, (255, 0, 0), "l2r")

            color = LEDMatrix.from_colormap(norm_output, "jet", color_space='hsv')
            color = (color[0], color[1], int(color[2] * 0.2))  # Dim the value for better visibility
            self.led.set_all(color, 'hsv')

            self.layer += 1
            time.sleep(2.0)

        print()
        print(f"Final value: {self.value.item()}")
        time.sleep(5.0)  # Keep the final state for a while before shutting down.

    def exchange_and_update(self):
        if self.pull_sock is None:
            raise RuntimeError("Runtime sockets not initialized; call setup_runtime_sockets() first.")
        # Send value to all neighbors (one PUSH socket per neighbor to ensure broadcast semantics).
        neighbors: List[Tuple[str, int]] = self.config["neighbors"]
        for (addr, port) in neighbors:
            if self.value.shape[1] <= 5:
                print(f"[{self.port}->{port}] h^{self.layer}={self.value}")
            else:
                print(
                    f"[{self.port}->{port}] h^{self.layer}=[... >5, mean={self.value.mean().item():.3f}]"
                )

            msg = {"layer": self.layer, "sender": self.port, "value": self.value}
            self.push_socks[(addr, port)].send(pickle.dumps(msg))

        # Receive values from all neighbors for this layer.
        needed = len(neighbors)
        while len(self.received[self.layer]) < needed:
            data = self.pull_sock.recv()  # blocking
            msg = pickle.loads(data)

            layer = int(msg["layer"])  # Could be from a different layer if out-of-sync, but protocol is synchronous
            sender = int(msg["sender"])
            value = msg["value"]

            if value.shape[1] <= 5:
                print(f"[{self.port}<-{sender}] h^{layer}={value}")
            else:
                print(f"[{self.port}<-{sender}] h^{layer}=[... >5, mean={value.mean().item():.3f}]")

            # Only store messages for the current layer
            if layer == self.layer:
                self.received[layer][sender] = value

        # Update value via the distributed GNN step
        self.value, self.output = self.distributed_model.update_node(
            self.layer, self.value, list(self.received[self.layer].values())
        )


def main():
    if len(sys.argv) != 2:
        node_index = "1"
    else:
        node_index = sys.argv[1]

    # Initialize the node and then start it.
    with NodeZMQ(node_index) as node:
        node.start()


if __name__ == "__main__":
    main()
