# Sandbox Branch

This `sandbox` branch provides a lightweight development environment for working with the `BLS` class without the need to run the full platform build process.

## Purpose

The full build process can be time-consuming, especially when testing small changes or new methods. This sandbox bypasses that by using dummy data stored locally (in `images/` and `annotations/` folders) and allows for direct interaction with the `BLS` class.

## Key Files

### `sandbox.py`

This is the entry point for sandbox development. It allows you to attach and test new methods to the `BLS` class without waiting on a Docker image to build.

**Example:**
```python
from bls import BLS

def new_method(self):
    pass

BLS.new_method = new_method

if __name__ == "__main__":
    bls = BLS()
    bls.new_method()
