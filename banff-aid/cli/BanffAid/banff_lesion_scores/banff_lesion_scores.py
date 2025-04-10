"""banff_lesion_scores.py.

This script provides functions to calculate several of the Banff lesion scores,
including ci (interstitial fibrosis), ct (tubular atrophy), cv (vascular
intimal thickening), as well as glomerulosclerosis proportions.


The functions rely on biopsy data—such as counts of total vs. sclerotic
glomeruli and estimated areas of fibrosis—and return standardized numeric
scores. Researchers and clinicians can use these results to evaluate
allograft status and track progression of chronic injury.

Usage:
    1. Import this script or call the functions directly to compute
       individual lesion scores:

         from banff_lesion_scores import calculate_gs, calculate_ci,
         calculate_cv, calculate_ct

         gs_score = calculate_gs(total_glomeruli, sclerotic_glomeruli)

    2. Integrate the lesion scores into broader diagnostic workflows.
       Reference the Banff classification guidelines for score
       interpretation and reporting.

Functions:
    - calculate_ci(UPDATE) -> float:
    - calculate_ct(UPDATE) -> float:
    - calculate_cv(UPDATE) -> float:
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

import json
from typing import Any

import numpy as np
from slicer_cli_web import CLIArgumentParser
from utils.utils import compute_max_distance, print_histogram, wilson_interval

#################################
## Functions to Compute Scores ##
#################################


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


def compute_ct(tubules: dict[str, Any]) -> dict[str, Any]:
    """Compute Banff lesion score ct for tubular atrophy.

    Args:
      tubules (dict[str, Any]):
        JSON annotations for tubules.

    Returns (dict[str, Any]):
        Information related to tubule diameters and the ct score.
    """
    tubule_elements = tubules["annotation"]["elements"]
    diameters: list[float] = []
    for te in tubule_elements:
        points = te["points"]
        max_distance = compute_max_distance(points)
        diameters.append(2 * abs(max_distance))

    # The "normal" diameter is considered to be the 80th percentile of the
    # data
    normal_diameter = np.percentile(diameters, 80)

    # Estimate the proportion of tubular atrophy
    atrophied_tubules = np.sum(np.asarray(diameters) < normal_diameter * 0.5)
    atrophy_proportion = atrophied_tubules / len(diameters)

    # Calculate ct score
    ct_score = 0
    if 0 < atrophy_proportion <= 0.25:
        ct_score = 1
    elif 0.25 < atrophy_proportion <= 0.5:
        ct_score = 2
    else:
        ct_score = 3

    # Get the range and IQR for the diameters
    range = f"[{round(np.min(diameters), 1)}, {round(np.max(diameters), 1)}]"
    iqr = (
        f"[{round(np.percentile(diameters, 25), 1)}, "
        f"{round(np.median(diameters), 1)}, "
        f"{round(np.percentile(diameters, 75), 1)}]"
    )
    print(f"quick: range = {range}")
    return {
        "Tubule Diameters": diameters,
        "Diameter Range": range,
        "Diameter IQR": iqr,
        "Proportion of Atrophy": round(atrophy_proportion, 4),
        "'ct' Score": ct_score,
    }


##############################
## Functions to Run Reports ##
##############################


def run_ci(configs: CLIArgumentParser) -> None:
    """Main."""
    # Load annotations using JSON
    print("\n\nREPORT FOR INTERSTITIAL FIBROSIS (BANFF LESION SCORE 'ci')")
    print("\n(No Implementation)\n")


def run_ct(configs: CLIArgumentParser) -> None:
    """Main."""
    print("\n\nREPORT FOR TUBULAR ATROPHY (BANFF LESION SCORE 'ct')\n")
    # Load annotations using JSON
    with open(configs.tubules_file) as file:
        tubules = json.load(file)

    # Compute ct
    ct = compute_ct(tubules)

    # Print report
    for key, value in ct.items():
        if key != "Tubule Diameters":
            print(f"{key}: {value}")
    print_histogram(ct["Tubule Diameters"])


def run_cv(configs: CLIArgumentParser) -> None:
    """Main."""
    # Load annotations using JSON
    print(
        "\n\nREPORT FOR VASCULAR INTIMAL THICKENING (BANFF LESION SCORE 'cv')"
    )
    print("\n(No Implementation)\n")


def run_gs(configs: dict[str, str]) -> None:
    """Main."""
    print("\n\nREPORT FOR GLOMERULOSCLEROSIS\n")
    # Load annotations using JSON
    with open(configs.non_gsg_file) as file:
        ngsg = json.load(file)
    with open(configs.gsg_file) as file:
        gsg = json.load(file)

    # Compute GS
    gs = compute_gs(ngsg, gsg)

    # At some point in the future, we will define the XML in a way that we
    # can save an actual report that can be downloaded for users. However,
    # we're not there yet, so we simply print out the results to the logger
    for key, value in gs.items():
        print(f"{key}: {value}")
