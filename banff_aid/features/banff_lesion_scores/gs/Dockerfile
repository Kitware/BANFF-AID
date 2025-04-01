# Start FROM the pre-built HistomicsTK image that includes
# large_image, slicer_cli_web, etc.
FROM dsarchive/histomicstk:latest
LABEL maintainer="Austin Allen <austin.allen@kitware.com>"

# Create a directory for your plugin code
ENV MY_PLUGIN_PATH=/gs
RUN mkdir -p $MY_PLUGIN_PATH
WORKDIR $MY_PLUGIN_PATH

# Copy plugin code. The '.' in COPY must be the directory
# that contains your plugin code on the host.
COPY . $MY_PLUGIN_PATH

# Install plugin (plus any extras). In many cases
# --find-links is used for large_image wheels.
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir . --find-links https://girder.github.io/large_image_wheels && \
    rm -rf /root/.cache/pip/*

# Switch to the CLI directory 
WORKDIR $MY_PLUGIN_PATH/cli   

# Optionally show that your new CLI is recognized
RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli

# If you want to see the help of your new CLI
RUN python -m slicer_cli_web.cli_list_entrypoint \
        glomerulosclerosis --help

# For demonstration, let's set the entrypoint to a shell
# that has your plugin code installed. You can adapt this
# to match how your environment typically runs.
ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
