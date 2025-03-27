"""glomerulosclerosis.py.

This script provides functions to calculate Glomerulosclerosis (GS), which is
the proportion of scarred glomeruli.


The functions rely on biopsy data—such as counts of total vs. sclerotic
glomeruli and estimated areas of fibrosis—and return standardized numeric
scores. Researchers and clinicians can use these results to evaluate
allograft status and track progression of chronic injury.

Usage:
    1. Import this script or call the functions directly to compute
       individual lesion scores:

         from banff_lesion_scores import calculate_gs

         gs_score = calculate_gs(total_glomeruli, sclerotic_glomeruli)

    2. Integrate the lesion scores into broader diagnostic workflows.
       Reference the Banff classification guidelines for score
       interpretation and reporting.

Functions:
    - calculate_gs(UPDATE) -> float:
        Calculates the proportion of glomeruli affected by sclerosis
        (glomerulosclerosis).

Reference:
    For further details on the Banff classification and lesion scoring
    criteria, consult the Banff Working Group publications and relevant
    nephropathology guidelines.

Disclaimer:
    These implementations are provided for research and educational
    purposes. Clinical decisions should be based on professional medical
    judgment and corroborated by multiple sources of information.

Author:
    Austin Allen, Kitware, Inc., 2025
"""

import argparse
import csv
import json
import os
from typing import Any


def compute_gs(
    non_globally_sclerotic_glomeruli: dict[str, Any],
    globally_sclerotic_glomeruli: dict[str, Any],
) -> dict[str, Any]:
    """Compute Glomerulosclerosis.

    Args:
    non_globally_sclerotic_glomeruli (): JSON annotations for NGSG.
    globally_sclerotic_glomeruli (): JSON annotations for GSG.

    Returns (float): Proportion of glomeruli that have global sclerosis.
    """
    count_ngsg = len(
        non_globally_sclerotic_glomeruli["annotation"]["elements"]
    )
    count_gsg = len(globally_sclerotic_glomeruli["annotation"]["elements"])

    # Compute proportion of GSG
    gsg_proportion = count_gsg / (count_gsg + count_ngsg)

    return {
        "Glomeruli Seen": count_gsg + count_ngsg,
        "Glomeruli Sclerosed #": count_gsg,
        "Glomeruli Sclerosed %": gsg_proportion,
    }


def configurations() -> dict[str, Any]:
    """CLI argument configurations."""
    parser = argparse.ArgumentParser(
        description="Glomerulosclerosis Configuration."
    )

    # Add arguments to the parser object
    parser.add_argument(
        "--annotation_directory",
        type=str,
        default=os.getcwd(),
        help=(
            "Directory path where glomeruli annotations are stored (default "
            "is current working directory)."
        ),
    )
    parser.add_argument(
        "--gsg_filename",
        type=str,
        default="globally_sclerotic_glomeruli.json",
        help=(
            "Name of the file containing annotations for globally sclerotic "
            "glomeruli."
        ),
    )
    parser.add_argument(
        "--ngsg_filename",
        type=str,
        default="non_globally_sclerotic_glomeruli.json",
        help=(
            "Name of the file containing annotations for non-globally "
            "sclerotic glomeruli (i.e. normal glomeruli)."
        ),
    )
    parser.add_argument(
        "--results_directory",
        type=str,
        default=os.getcwd(),
        help=(
            "Name of the directory in which to store results. Results will be "
            "saved as 'glomerulosclerosis.csv.'"
        ),
    )

    # Create file paths
    args = parser.parse_args()
    ngsg_path = f"{args.annotation_directory}/{args.ngsg_filename}"
    gsg_path = f"{args.annotation_directory}/{args.gsg_filename}"

    return {
        "ngsg_path": ngsg_path,
        "gsg_path": gsg_path,
        "results_directory": args.results_directory,
    }


def main(configs: dict[str, str]) -> None:
    """Main."""
    # Load annotations using JSON
    with open(configs["ngsg_path"]) as file:
        ngsg = json.load(file)
    with open(configs["gsg_path"]) as file:
        gsg = json.load(file)

    # Compute GS
    gs = compute_gs(ngsg, gsg)
    print(gs)

    # Save results to results-folder location (defaults to CWD)
    with open(
        f"{configs['results_directory']}/glomerulosclerosis.csv",
        "w",
        newline="",
    ) as file:
        # Use the CSV dictionary writer to store the results in the specified
        # directory
        writer = csv.DictWriter(file, fieldnames=gs.keys())
        writer.writeheader()
        writer.writerow(gs)


if __name__ == "__main__":
    configs = configurations()
    main(configs)
