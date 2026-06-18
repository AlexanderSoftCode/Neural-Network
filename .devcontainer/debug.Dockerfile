FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install ONLY Python 3.14 and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    python3.14 \
    python3.14-dev \
    && rm -rf /var/lib/apt/lists/*


# 3. Download and install pip using Python 3.14
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.14

# 4. Verify the installation succeeded
RUN python3.14 -m pip --version