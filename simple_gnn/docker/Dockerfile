# Use official Python base image (multi-arch: amd64, arm64)
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        avahi-daemon \
        dbus \
        git \
        build-essential \
        tmux \
        curl \
        openssh-client \
        iputils-ping \
        libnss-mdns \
        scons \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install torch-geometric
RUN pip3 install pyyaml matplotlib networkx tqdm codetiming
RUN pip3 install rpi_ws281x

# Set up RGB shield
WORKDIR /workspace
RUN git clone https://github.com/jgarff/rpi_ws281x.git
WORKDIR /workspace/rpi_ws281x

RUN sed -i 's/^#define WIDTH.*/#define WIDTH                   8/' main.c && \
    sed -i 's/^#define HEIGHT.*/#define HEIGHT                  4/' main.c
RUN scons

# Clone git repositories
WORKDIR /workspace
RUN git clone https://github.com/mkrizmancic/my_graphs_dataset.git
RUN git clone https://github.com/mkrizmancic/distributed_computation.git

# Build and install
RUN cd /workspace/my_graphs_dataset && pip3 install -e .

RUN apt-get update && apt-get install -y libnss-mdns

# Default command
CMD ["sh", "-c", "service dbus start && /usr/sbin/avahi-daemon --daemonize && exec /bin/bash"]
