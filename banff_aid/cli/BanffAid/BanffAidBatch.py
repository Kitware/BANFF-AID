"""BANFF-AID CLI for batch processing."""
import importlib
import os
import subprocess
import sys

import pandas as pd
from slicer_cli_web import CLIArgumentParser
from tqdm import tqdm

PYTHON = sys.executable
BanffAid = importlib.util.find_spec('BanffAid').origin


def main():
    args = CLIArgumentParser().parse_args()

    file_list = []
    for filename in os.listdir(args.images_dir):
        if filename.endswith(".svs") or filename.endswith(".scn"):
            xml_file = os.path.join(args.images_dir, filename)[:-4] + ".xml"
            if os.path.exists(xml_file):
                file_list.append(filename)

    failures = 0
    dfs = []
    for filename in tqdm(file_list, desc="WSIs", position=0):
        image_file = os.path.join(args.images_dir, filename)
        print(image_file)
        params = f'--image-filepath "{image_file}" --results-folder "{args.results_folder}"'
        command = f'{PYTHON} {BanffAid} {params}'

        proc = subprocess.run(command, shell=True, cwd=args.results_folder, check=False)
        if proc.returncode:
            print(f'Command failed with exit code {proc.returncode}: {proc.args!r}')
            failures += 1
        else:
            results_table = os.path.join(args.results_folder, filename + '_report.csv')
            dfs.append(pd.read_csv(results_table))

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(args.results_folder, '_batch_report.csv'), index=False)
    if failures > 0:
        print(f'Batch processing completed with {failures} failures.')


if __name__ == "__main__":
    main()
