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

# import csv
import json
import math
from typing import Any

from slicer_cli_web import CLIArgumentParser


def compute_gs(
    non_globally_sclerotic_glomeruli: dict[str, Any],
    globally_sclerotic_glomeruli: dict[str, Any],
) -> dict[str, Any]:
    """Compute Glomerulosclerosis.

    Args:
      non_globally_sclerotic_glomeruli (dict[str, Any]): JSON annotations for
      NGSG.
      globally_sclerotic_glomeruli (dict[str, Any]): JSON annotations for GSG.

    Returns (float):
      Proportion of glomeruli that have global sclerosis.
    """
    count_ngsg = len(
        non_globally_sclerotic_glomeruli["annotation"]["elements"]
    )
    count_gsg = len(globally_sclerotic_glomeruli["annotation"]["elements"])
    n = count_ngsg + count_gsg

    # Compute proportion of GSG
    gsg_proportion = count_gsg / n

    # Compute 95% confidence interval
    lower_bound, upper_bound = wilson_interval(count_gsg, n)
    confidence_interval = f"[{round(lower_bound, 4)}, {round(upper_bound, 4)}]"

    return {
        "Glomeruli Seen": n,
        "Glomeruli Sclerosed #": count_gsg,
        "Glomeruli Sclerosed %": round(gsg_proportion, 4),
        "95% Confidence Interval": confidence_interval,
    }


def wilson_interval(k: int, n: int) -> tuple[float, float]:
    """Compute 95% Wilson score confidence interval for a binomial proportion.

    Args:
        k (int): Number of successes (e.g., number of sclerosed glomeruli)
        n (int): Total number of trials (e.g., total glomeruli)

    Returns:
        (float, float): A tuple containing the lower and upper bounds of the
        confidence interval.

    Raises:
        ValueError: If n is zero.

    Reference:
        https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    """
    if n == 0:
        raise ValueError(
            "Total number of trials (n) must be greater than zero."
        )

    # Compute the sample proportion.
    p_hat = k / n

    # For a 95% confidence interval, z is typically 1.96.
    z = 1.96

    # Adjusted denominator.
    denominator = 1 + (z**2 / n)

    # Adjusted center.
    center_adjusted = p_hat + (z**2 / (2 * n))

    # The adjustment term.
    adjustment = z * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))

    # Compute lower and upper bounds.
    lower_bound = (center_adjusted - adjustment) / denominator
    upper_bound = (center_adjusted + adjustment) / denominator

    return lower_bound, upper_bound


def main(configs: dict[str, str]) -> None:
    """Main."""
    print(configs)
    # Load annotations using JSON
    with open(configs.non_gsg_file) as file:
        ngsg = json.load(file)
    with open(configs.gsg_file) as file:
        gsg = json.load(file)

    # Compute GS
    gs = compute_gs(ngsg, gsg)

    # At some point in the future, I will have define the XML in a way that 
    # we can save an actual report that can be downloaded for users. However,
    # we're not there yet, so I'm just going to print out the results to the 
    # logger

    # Print report
    bar = "#" * 42 + "\n"
    title = " " * 18 + "REPORT" + " " * 18 + "\n"
    print(bar + title)
    for key, value in gs.items():
        print(f"{key}: {value}")
    print(bar)

    # # Save results to results-folder location (defaults to CWD)
    # with open(
    #     f"{configs['results_directory']}/glomerulosclerosis.csv",
    #     "w",
    #     newline="",
    # ) as file:
    #     # Use the CSV dictionary writer to store the results in the specified
    #     # directory
    #     writer = csv.DictWriter(file, fieldnames=gs.keys())
    #     writer.writeheader()
    #     writer.writerow(gs)


if __name__ == "__main__":
    configs = CLIArgumentParser().parse_args()
    main(configs)
