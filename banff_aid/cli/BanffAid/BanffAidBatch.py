"""BANFF-AID CLI for batch processing."""
import importlib
import os
import subprocess
import sys

from slicer_cli_web import CLIArgumentParser
from tqdm import tqdm

PYTHON = sys.executable
BanffAid = importlib.util.find_spec('BanffAid').origin


def main():
    args = CLIArgumentParser().parse_args()

    file_list = []
    for filename in os.listdir(args.images_dir):
        if filename.endswith(".svs") or filename.endswith(".scn"):
            image_file = os.path.join(args.images_dir, filename)
            xml_file = image_file[:-4] + ".xml"
            if os.path.exists(xml_file):
                file_list.append(image_file)

    failures = 0
    for image_file in tqdm(file_list, desc="WSIs", position=0):
        print(image_file)
        params = f'--image-filepath "{image_file}" --results-folder "{args.results_folder}"'
        command = f'{PYTHON} {BanffAid} {params}'

        proc = subprocess.run(command, shell=True, cwd=args.results_folder, check=False)
        if proc.returncode:
            print(f'Command failed with exit code {proc.returncode}: {proc.args!r}')
            failures += 1

    if failures > 0:
        print(f'Batch processing completed with {failures} failures.')


if __name__ == "__main__":
    main()
