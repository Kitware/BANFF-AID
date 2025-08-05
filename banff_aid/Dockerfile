# Start FROM the pre-built HistomicsTK image that includes
# large_image, slicer_cli_web, etc.
FROM dsarchive/histomicstk:latest
LABEL maintainer="Dženan Zukić <dzenan.zukic@kitware.com>"

# Create a directory for your plugin code
ENV MY_PLUGIN_PATH=/plugin
RUN mkdir -p $MY_PLUGIN_PATH
WORKDIR $MY_PLUGIN_PATH

# Install libgl1 for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy the entire repository (now including pyproject.toml) into the container.
# When building, set the context to the repository root.
# COPY pyproject.toml $MY_PLUGIN_PATH/
COPY . $MY_PLUGIN_PATH

# Upgrade pip and setuptools and install the package using pyproject.toml as source
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir . --find-links https://girder.github.io/large_image_wheels && \
    rm -rf /root/.cache/pip/* && apt-get update && apt-get install -y gcc python3-dev git &&\
    rm -rf /var/lib/apt/lists/*

WORKDIR $MY_PLUGIN_PATH/banff_aid/cli

# Ensure your new CLI is recognized
RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli

# If you want to see the help of your new CLI
RUN python -m slicer_cli_web.cli_list_entrypoint BanffAid --help

# Set the entrypoint to a shell that has your plugin code installed
ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]

