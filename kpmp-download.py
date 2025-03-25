"""Python script to run Bash scripts to download data from KPMP atlas.

Usage:
python3 kpmp-download.py {files} {target_dir}
    - files: The path to the CSV containing all the filenames for download
    - target_dir: The directory in which to save the downloaded files
"""

import argparse
import os
import subprocess

import pandas as pd
from tqdm import tqdm


def download_files(
    file_names: list[str],
    internal_id: list[str],
    target_dir: str,
) -> None:
    """Downloads all files in 'filenames' list from the KPMP data repo.

    file_names (list[str]): All files to be downloaded from KPMP data repo.
    internal_id (list[str]): All internal IDs (sub-directories) for the files
    to be downloaded.
    target_dir (str): The local target directory for all downloaded files.
    """
    BASE_URL="https://atlas.kpmp.org/api/v1/file/download"

    for i in tqdm(range(len(file_names))):
        file_url=f"{BASE_URL}/{internal_id[i]}/{file_names[i]}"
        file_path=f"{target_dir}/{file_names[i]}"

        # Run the script and check that the result was successful
        result = subprocess.run(["curl", "-o", file_path, file_url])
        if result.returncode == 0:
            print("Download successful!")
        else:
            print(f"Download failed with error: {result.stderr}")


# Set up argparse to handle command-line arguments
parser = argparse.ArgumentParser(
    description="Download a file using the KPMP API."
)
parser.add_argument(
    "files", help="The path to the CSV containing all filenames."
)
parser.add_argument(
    "target_dir", help="The directory to save the downloaded file."
)

# Parse the arguments
args = parser.parse_args()

# Expand the target directory if it contains '~'
target_dir = os.path.expanduser(args.target_dir)

# Retrieve the list of filenames to be downloaded from the KPMP data repo. Each
# file has an internal package ID that acts as a subdirectory for the file in the
# download repo
files = pd.read_csv(args.files)
file_names = files["File Name"]
internal_id = files["Internal Package ID"]


if __name__ == "__main__":
    download_files(file_names, internal_id, target_dir)
