"""Sandbox for development."""

from bls import BLS


def new_method(self):
    pass


BLS.new_method = new_method

if __name__ == "__main__":
    bls = BLS()
    bls.new_method()
