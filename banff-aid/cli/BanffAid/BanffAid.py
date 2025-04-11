"""BanffAid.py.

This script runs all Banff lesion score functions to generate a report.

The functions rely on biopsy data related to tubules and glomeruli, and they
returns standardized numeric scores. Researchers and clinicians can use these
results to evaluate allograft status and track progression of chronic injury.

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
    run_ci,
    run_ct,
    run_cv,
    run_gs,
)
from slicer_cli_web import CLIArgumentParser


def main() -> None:
    """Main."""
    configs = CLIArgumentParser().parse_args()
    run_ci(configs)
    run_ct(configs)
    run_cv(configs)
    run_gs(configs)


if __name__ == "__main__":
    main()
