"""BANFF-AID CLI entry point for computing Banff lesion scores from biopsy data.

This script runs all Banff lesion score functions to generate a structured
PDF report. The scoring functions rely on biopsy annotations for tubules,
glomeruli, arteries, and interstitial regions, and return standardized
numeric scores. These results help researchers and clinicians evaluate
allograft status and monitor progression of chronic injury.

Reference:
    For details on the Banff classification and scoring criteria, consult
    the Banff Working Group publications and relevant nephropathology guidelines.
"""

from banff_lesion_score.banff_lesion_score import BanffLesionScore
from slicer_cli_web import CLIArgumentParser


def main() -> None:
    """Parse CLI arguments and run the BANFF-AID lesion scoring pipeline.

    This function uses Slicer CLI Web's argument parser to extract runtime
    inputs, initializes the BanffLesionScore object, and executes the full
    report generation and upload process.

    Returns:
        None
    """
    args = CLIArgumentParser().parse_args()
    bls = BanffLesionScore(args)
    bls.main()


if __name__ == "__main__":
    main()
