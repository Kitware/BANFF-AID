# Sandbox Branch

This `sandbox` branch provides a lightweight development environment for working with the `BLS` class without the need to run the full platform build process.

## Purpose

The full build process can be time-consuming, especially when testing small changes or new methods. This sandbox bypasses that by using dummy data stored locally and allows for direct interaction with the `BLS` class.

The BLS class expects a `data/` directory nested under `sandbox/`, where annotations and a large image are stored. All annotations are expected to be in `data/annotations/`. You will need to create this directory yourself. For developers associated with Kitware, you can use the data stored [here](https://drive.google.com/drive/folders/10ne8lochBjKI7ETjtSrUF2lk5OnyNnFk).

## Key Files

### `sandbox.py`

This is the entry point for sandbox development. It allows you to attach and test new methods to the `BLS` class without waiting on a Docker image to build.

### `bls.py`

You may need to modify the class __init__ method, depending on your development needs. For starters, consider:
- Matching the Girder API URL for your instance of HistomicsTK
- The username and password associated with your account on HistomicsTK
- The Girder ID for the image you will be processing
- The Girder ID for the folder in which you want results stored


**Example:**
```python
from bls import BLS

def new_method(self):
    pass

BLS.new_method = new_method

if __name__ == "__main__":
    bls = BLS()
    bls.new_method()
