"""BanffAid.py.

This script runs all Banff lesion score functions to generate a report.

The functions rely on biopsy data related to tubulses, and it returns
standardized numeric scores. Researchers and clinicians can use these results
to evaluate allograft status and track progression of chronic injury.

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

# from utils.utils import convert_notebook_to_pdf


def main() -> None:
    """Main."""
    # report_file = "report/report.ipynb"
    # convert_notebook_to_pdf(report_file)
    configs = CLIArgumentParser().parse_args()
    print("starting main")
    run_ci(configs)
    print("ci complete")
    run_ct(configs)
    print("ct complete")
    run_cv(configs)
    print("cv complete")
    run_gs(configs)
    print("gs complete")


if __name__ == "__main__":
    main()
    # # Get the directory of the current file
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # # Build the absolute path to the notebook
    # report_file = os.path.join(base_dir, "report", "BanffReport.ipynb")

    # # Check if the file exists
    # if not os.path.exists(report_file):
    #     raise FileNotFoundError(f"Notebook file not found: {report_file}")

    # convert_notebook_to_pdf(report_file)
