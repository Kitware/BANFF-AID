"""BANFF-AID plugin core for computing Banff lesion scores from WSI annotations.

This module defines the BanffLesionScore class, which processes whole slide
image annotations from Girder/DSA to compute quantitative Banff lesion scores
for renal biopsy analysis. It includes logic for:

- Fetching and validating named annotations from Girder
- Measuring anatomical structures (e.g., tubule diameters, glomeruli counts)
- Calculating Banff lesion scores including ci, ct, cv, and gs
- Rendering structured PDF reports with text, tables, and visual plots
- Uploading the final report to a specified Girder folder

The BanffLesionScore class serves as the primary engine used in the BANFF-AID
CLI plugin and can be run as a standalone report generator in HistomicsTK.
"""

from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from girder_client import GirderClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas
from slicer_cli_web import CLIArgumentParser
from utils.utils import (
    compute_max_distance,
    convert_to_microns,
    create_table,
    draw_plot,
    draw_table,
    draw_text,
    fetch_annotations,
    fetch_mpp,
    wilson_interval,
)


class BanffLesionScore:
    """Compute Banff lesion scores and generate PDF reports from WSI annotations.

    This class implements the core analysis pipeline for the BANFF-AID plugin.
    It fetches structured annotations from a Girder/DSA instance, computes
    lesion-specific Banff scores, and renders a full PDF report containing
    summary tables, histograms, and textual interpretations.

    Attributes:
        image_id (str): Girder file ID of the whole slide image.
        results_folder (str): ID of the folder where the PDF report will be uploaded.
        gc (GirderClient): Authenticated Girder client for interacting with the API.
        <annotation name>_annotation (dict | None): Each annotation fetched from Girder
        corresponding to a Banff lesion component (e.g., tubules, glomeruli, arteries).

    Methods:
        compute_ci(): Compute score for interstitial fibrosis (ci).
        compute_ct(): Compute score for tubular atrophy (ct).
        compute_cv(): Compute score for vascular intimal thickening (cv).
        compute_gs(): Compute glomerulosclerosis score with confidence intervals.
        draw_<score>(): Render visual report sections for each lesion.
        generate_report(): Generate and save the full PDF report.
        main(): Run the full pipeline and upload the report to Girder.
    """

    def __init__(self, args: CLIArgumentParser) -> None:
        """Initialize the BanffLesionScore pipeline with CLI arguments.

        This constructor sets up the image and results folder IDs, authenticates
        with the Girder API, and fetches all required annotations by name. The
        annotation data is stored for later use by scoring and reporting methods.

        Args:
            args (CLIArgumentParser): Parsed CLI arguments, including Girder
            credentials, image file ID, annotation filenames, and results folder ID.
        """
        # Image and folder
        self.image_id = args.image_id
        self.results_folder = args.results_folder

        # Girder Client Instantiation
        self.gc = GirderClient(apiUrl=args.girder_api_url)
        self.gc.authenticate(args.username, args.password)

        # Annotations
        annotations = fetch_annotations(self.gc, args)
        self.non_gsg_annotation = annotations[args.non_gsg_filename]
        self.gsg_annotation = annotations[args.gsg_filename]
        self.tubules_annotation = annotations[args.tubules_filename]
        self.arteries_annotation = annotations[args.arteries_filename]
        self.cortical_interstitium_annotation = annotations[
            args.cortical_interstitium_filename
        ]
        self.medullary_interstitium_annotation = annotations[
            args.medullary_interstitium_filename
        ]

    def compute_ci(self) -> None:
        """Compute the interstitial fibrosis (ci) Banff lesion score.

        This method is a placeholder for future implementation. It is expected to
        analyze cortical interstitial regions and calculate the extent of fibrosis
        based on annotation data.

        Returns:
            None
        """
        return ["No implementation."]

    def compute_ct(self) -> dict[str, Any]:
        """Compute the tubular atrophy (ct) Banff lesion score.

        This method analyzes annotated renal tubules and calculates their
        diameters in microns, using image resolution metadata (microns per pixel).
        The "normal" tubule diameter is defined as the 80th percentile of the
        observed diameters. Tubules with diameters less than 50% of this reference
        are considered atrophied.

        The ct score is assigned based on the proportion of atrophied tubules:
            - Score 1: 0–25% atrophied
            - Score 2: 25–50% atrophied
            - Score 3: >50% atrophied

        Returns:
            dict[str, Any]: A dictionary of quantitative results including raw
            diameters, range, interquartile range (IQR), atrophy percentage, and
            the computed ct score.
        """
        # First, we need the microns per pixel (x and y) so we can have
        # tubule diameters in microns
        mpp_x, mpp_y = fetch_mpp(self.gc, self.image_id)

        # Now we calculate the diameter as the maximum
        tubule_elements = self.tubules_annotation["annotation"]["elements"]
        diameters: list[float] = []
        for element in tubule_elements:
            points = element["points"]
            points_um = convert_to_microns(points, mpp_x, mpp_y)
            max_distance = compute_max_distance(points_um)
            diameters.append(2 * abs(max_distance))

        # The "normal" diameter is considered to be the 80th percentile of the
        # data
        normal_diameter = np.percentile(diameters, 80)

        # Estimate the proportion of tubular atrophy
        atrophied_tubules = np.sum(
            np.asarray(diameters) < normal_diameter * 0.5
        )
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
        range = (
            f"[{round(np.min(diameters), 1)}, {round(np.max(diameters), 1)}]"
        )
        iqr = (
            f"[{round(np.percentile(diameters, 25), 1)}, "
            f"{round(np.median(diameters), 1)}, "
            f"{round(np.percentile(diameters, 75), 1)}]"
        )

        # Combine results into a summary
        return {
            "Tubule Diameters": diameters,
            "Tubules Seen": len(diameters),
            "Diameter Range": range,
            "Diameter IQR": iqr,
            "Diamter Length at 80th Percentile": round(normal_diameter, 2),
            "Atrophy %": round(100 * atrophy_proportion, 2),
            "'ct' Score": ct_score,
        }

    def compute_cv(self) -> None:
        """Compute the vascular intimal thickening (cv) Banff lesion score.

        This method is currently unimplemented. It is intended to analyze
        arterial wall annotations and compute a cv score based on the degree
        of intimal thickening observed.

        Returns:
            None
        """
        return ["No implementation."]

    def compute_gs(self) -> dict[str, Any]:
        """Compute the glomerulosclerosis (gs) Banff lesion score.

        This method counts the number of sclerosed and non-sclerosed glomeruli
        from annotated regions and calculates the proportion of sclerosed glomeruli.
        A 95% Wilson score confidence interval is also computed to quantify
        uncertainty in the proportion estimate.

        Returns:
            dict[str, Any]: A dictionary containing the total glomeruli count,
            number and percentage sclerosed, and the 95% confidence interval
            as a stringified range.
        """
        count_ngsg = len(self.non_gsg_annotation["annotation"]["elements"])
        count_gsg = len(self.gsg_annotation["annotation"]["elements"])
        n = count_ngsg + count_gsg

        # Compute proportion of GSG
        gsg_proportion = count_gsg / n
        glomeruli_sclerosed_percentage = f"{100 * round(gsg_proportion, 4)}"

        # Compute 95% confidence interval
        lower_bound, upper_bound = wilson_interval(count_gsg, n)
        confidence_interval = (
            f"[{100 * round(lower_bound, 4)}, {100 * round(upper_bound, 4)}]"
        )

        return {
            "Glomeruli Seen": n,
            "Glomeruli Sclerosed #": count_gsg,
            "Glomeruli Sclerosed %": glomeruli_sclerosed_percentage,
            "95% Confidence Interval For GS %": confidence_interval,
        }

    def draw_ci(
        self, pdf_canvas: Canvas, x: float, y: float
    ) -> tuple[Canvas, float]:
        """Draw interstitial fibrosis (ci) section on the PDF report.

        This method calls the compute_ci() function and formats its results as
        a list of lines. The content is then drawn on the provided ReportLab
        canvas at the specified (x, y) position.

        Args:
            pdf_canvas (Canvas): The ReportLab canvas to draw on.
            x (float): The x-coordinate where text should begin.
            y (float): The current y-coordinate for text placement.

        Returns:
            tuple[Canvas, float]: The canvas and the updated y-coordinate after drawing.
        """
        ci_results = self.compute_ci()
        ci_content = ["Interstitial Fibrosis:"]
        for item in ci_results:
            ci_content.append(item)

        pdf_canvas, y = draw_text(pdf_canvas, x, y, ci_content)

        return pdf_canvas, y

    def draw_ct(
        self, pdf_canvas: Canvas, x: float, y: float
    ) -> tuple[Canvas, float]:
        """Draw tubular atrophy (ct) section on the PDF report.

        This method calls compute_ct() to calculate atrophy metrics and then
        renders two components on the report:
            - A summary table of key ct metrics (e.g., IQR, atrophy %, score)
            - A histogram of tubule diameters with an overlaid atrophy threshold

        The content is drawn sequentially on the provided canvas, and the
        y-coordinate is updated after each visual element.

        Args:
            pdf_canvas (Canvas): The ReportLab canvas to draw on.
            x (float): The x-coordinate where the table and plot should begin.
            y (float): The current y-coordinate for placement.

        Returns:
            tuple[Canvas, float]: The updated canvas and y-coordinate after drawing.
        """
        ct_results = self.compute_ct()
        ct_text = ["Tubular Atrophy:"]

        # Create summary table
        diameters = ct_results.pop("Tubule Diameters", None)
        ct_table = create_table(ct_results)
        pdf_canvas, y = draw_table(pdf_canvas, ct_table, x, y, ct_text)

        # Create histogram for diameters
        fig = plt.figure(figsize=(3, 2))
        plt.title("Tubule Diameters")
        plt.hist(diameters)
        plt.ylabel("Count")
        plt.xlabel("Diameter (microns)")
        ymin, ymax = plt.ylim()
        threshold = (
            ct_results["Diamter Length at 80th Percentile"]
            - 0.5 * ct_results["Diamter Length at 80th Percentile"]
        )
        plt.vlines(
            x=threshold,
            ymin=ymin,
            ymax=ymax + 50,
            label=f"Atrophy Threshold = {round(threshold, 1)}",
            colors="#b22222",
        )
        plt.legend()

        pdf_canvas, y = draw_plot(pdf_canvas, fig, x, y)

        return pdf_canvas, y

    def draw_cv(
        self, pdf_canvas: Canvas, x: float, y: float
    ) -> tuple[Canvas, float]:
        """Draw vascular intimal thickening (cv) section on the PDF report.

        This method calls compute_cv() and formats the returned results as a
        block of text. The output is drawn onto the provided ReportLab canvas
        at the specified (x, y) position.

        Args:
            pdf_canvas (Canvas): The canvas to draw the text on.
            x (float): The starting x-coordinate for the text block.
            y (float): The current y-coordinate to begin drawing.

        Returns:
            tuple[Canvas, float]: The canvas and the updated y-coordinate after drawing.
        """
        cv_results = self.compute_cv()
        cv_text = ["Vascular Intimal Thickening:"]
        for item in cv_results:
            cv_text.append(item)

        pdf_canvas, y = draw_text(pdf_canvas, x, y, cv_text)

        return pdf_canvas, y

    def draw_gs(
        self, pdf_canvas: Canvas, x: float, y: float
    ) -> tuple[Canvas, float]:
        """Draw glomerulosclerosis (gs) section on the PDF report.

        This method calls compute_gs() to retrieve counts, percentages, and
        confidence intervals related to glomerular sclerosis. The results are
        formatted into a summary table and rendered on the PDF canvas at the
        given (x, y) position.

        Args:
            pdf_canvas (Canvas): The canvas to draw the table on.
            x (float): The x-coordinate for table placement.
            y (float): The current y-coordinate for drawing.

        Returns:
            tuple[Canvas, float]: The updated canvas and y-coordinate after drawing.
        """
        gs_results = self.compute_gs()
        gs_text = ["Glomerulosclerosis:"]
        gs_table = create_table(gs_results)

        pdf_canvas, y = draw_table(pdf_canvas, gs_table, x, y, gs_text)

        return pdf_canvas, y

    def generate_report(self) -> str:
        """Generate and save the full PDF report of Banff lesion scores.

        This method sequentially renders each Banff lesion score section
        (ci, ct, cv, gs) onto a PDF using precomputed annotations. Each section
        may include summary text, tables, and plots, depending on the lesion type.

        A report header with the title and timestamp is added, and each section is
        drawn in order using its corresponding draw_* method. The final PDF is
        saved to disk and the file path is returned.

        Returns:
            str: The file path of the generated PDF report.
        """
        # Set title and timestamp for report
        date = datetime.now()
        report_title = [
            "BANFF-AID",
            "Banff Lesion Scores Report",
            date.strftime("Report Timestamp: %Y-%m-%d %H:%M:%S"),
        ]

        # Create Canvas object and begin drawing
        x, y = 50, letter[1] - 50
        path = "BANFF-AID Report.pdf"
        pdf_canvas = Canvas(path, pagesize=letter)
        pdf_canvas, y = draw_text(
            pdf_canvas, x, y, report_title, centered=True, font_size=16
        )

        # Begin adding results from each of the Banff lesion scores

        # Interstitial Fibrosis
        pdf_canvas, y = self.draw_ci(pdf_canvas, x, y)

        # Tubular Atrophy
        pdf_canvas, y = self.draw_ct(pdf_canvas, x, y)

        # Vascular Intimal Thickening
        pdf_canvas, y = self.draw_cv(pdf_canvas, x, y)

        # Glomerulosclerosis
        pdf_canvas, y = self.draw_gs(pdf_canvas, x, y)

        pdf_canvas.save()

        return path

    def main(self) -> None:
        """Run the full BANFF-AID pipeline and upload the PDF report to Girder.

        This method calls generate_report() to compute Banff lesion scores and
        render the final report. The resulting PDF is then uploaded to the
        specified Girder results folder using the authenticated client.

        Returns:
            None
        """
        # Generate the report and upload it to the output folder
        report_path = self.generate_report()
        self.gc.uploadFileToFolder(self.results_folder, report_path)
