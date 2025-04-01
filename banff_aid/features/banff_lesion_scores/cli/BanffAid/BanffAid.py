"""ct.py.

This script provides functions to calculate ct, which is used to assess 
tubular atrophy.


The functions rely on biopsy data related to tubulses, and it returns 
standardized numeric scores. Researchers and clinicians can use these results 
to evaluate allograft status and track progression of chronic injury.

Usage:
    1. Import this script or call the functions directly to compute
       individual lesion scores:

         from banff_lesion_scores import calculate_ct

         ct_score = calculate_ct(tubules)

    2. Integrate the lesion scores into broader diagnostic workflows.
       Reference the Banff classification guidelines for score
       interpretation and reporting.

Functions:
    - calculate_ct(tubules) -> int:
        Calculates the ct score to assess tubular atrophy

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

from banff_lesion_scores.banff_lesion_scores import (
    run_ci, run_ct, run_cv, run_gs
)
from slicer_cli_web import CLIArgumentParser


def main(configs: CLIArgumentParser) -> None:
    """Main."""
    run_ci(configs)
    run_ct(configs)
    run_cv(configs)
    run_gs(configs)


if __name__ == "__main__":
    configs = CLIArgumentParser().parse_args()
    main(configs)

    