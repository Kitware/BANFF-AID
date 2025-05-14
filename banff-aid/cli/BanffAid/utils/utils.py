"""Utility functions for BANFF-AID renal pathology plugin.

This module provides core helper functions used throughout the BANFF-AID
pipeline, including geometric computations, annotation parsing, metadata
extraction, statistical scoring, and PDF layout utilities.

Functions include:
- Annotation processing: fetching annotations, validating uniqueness
- Slide metadata extraction: microns-per-pixel (MPP) retrieval from Girder
- Measurement utilities: pixel-to-micron conversion, max interior distance
- Scoring support: Wilson confidence intervals for proportions
- Report generation: text layout, plot rendering, and table drawing on PDF canvas

These functions are used by the BANFF-AID Slicer CLI plugin to compute Banff
lesion scores, visualize pathology results, and generate a structured PDF
report for renal biopsy whole slide images.
"""

import math
import textwrap
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from girder_client import GirderClient
from matplotlib.figure import Figure
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas
from sklearn.neighbors import KDTree
from slicer_cli_web import CLIArgumentParser


def ci_threshold(proportion: float, discrete: bool = True) -> float | int:
    """Compute a Banff interstitial fibrosis (ci) score from a proportion.

    Depending on the `discrete` flag, returns either a discrete score (0-3)
    based on fixed cutoffs, or a continuous score (0.0-4.0) by linear
    interpolation within those bins.

    Args:
        proportion (float): Fraction of fibrotic tissue (0.0-1.0).
        discrete (bool): If True, return one of {0, 1, 2, 3}. Otherwise,
            return a float in [0.0, 4.0).

    Returns:
        int | float: Discrete integer score if `discrete` is True,
            else a continuous float score.
    """
    if discrete:
        if proportion < 0.06:
            return 0
        elif proportion < 0.26:
            return 1
        elif proportion < 0.51:
            return 2
        else:
            return 3
    else:
        if proportion < 0.06:
            return (proportion / 0.06) * 1
        elif proportion < 0.26:
            return 1 + ((proportion - 0.06) / (0.26 - 0.06)) * 1
        elif proportion < 0.51:
            return 2 + ((proportion - 0.26) / (0.51 - 0.26)) * 1
        else:
            return 3 + ((proportion - 0.51) / (1 - 0.51)) * 1


def compute_fibrosis_area(edge: float, cutoff: float) -> tuple[float, float]:
    """Compute normal and fibrotic area from edge length and cutoff.

    Treat each edge as defining a square region. If the edge length exceeds
    the cutoff, the normal area is the cutoff² and the excess is fibrosis.
    Otherwise, the entire square is normal and fibrosis is zero.

    Args:
        edge (float): Edge length of the square region.
        cutoff (float): Cutoff edge length defining normal tissue.

    Returns:
        tuple[float, float]: A pair (normal_area, fibrosis_area).
    """
    if edge**2 > cutoff**2:
        normal = cutoff**2
        fibrosis = edge**2 - cutoff**2
    else:
        normal = edge**2
        fibrosis = 0

    return normal, fibrosis


def compute_max_distance(points):
    """Compute the maximum distance from an interior point to the edge of a polygon.

    Given a list of (x, y, z) points defining a polygon, this function creates a
    binary mask and computes the Euclidean distance transform to find the
    furthest interior point from the edge. This is useful for estimating the radius
    of roughly circular or elliptical structures like renal tubules.

    Args:
        points (list[list[float]]): A closed set of ordered points in (x, y, z)
        format, typically from a Girder annotation polygon.

    Returns:
        float: The negative of the maximum Euclidean distance from any
        interior pixel to the edge of the polygon.
    """
    # Convert the points to a NumPy array of type int32
    points_np = np.array(points, dtype=np.int32)

    # Determine the bounding box of the polygon
    min_x = points_np[:, 0].min()
    min_y = points_np[:, 1].min()
    max_x = points_np[:, 0].max()
    max_y = points_np[:, 1].max()

    # Compute width and height for the mask (include boundaries)
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    # Create a blank image (mask) using Pillow (mode 'L' for grayscale)
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    # Shift polygon coordinates so that they fit within the mask
    adjusted_points = [(p[0] - min_x, p[1] - min_y) for p in points]

    # Fill the polygon in the mask with white (value 1)
    draw.polygon(adjusted_points, outline=1, fill=1)

    # Convert the PIL image to a NumPy array
    mask = np.array(mask_img)

    # Compute the Euclidean distance transform on the mask
    dist_transform = ndi.distance_transform_edt(mask)

    # The maximum distance from the interior to the edge
    max_distance = dist_transform.max()

    # Return the maximum distance
    return -max_distance


def compute_polygon_centroid(polygon: list[list[float]]) -> tuple[float, float]:
    """Compute the centroid of a 2D polygon via the shoelace method.

    Calculates centroid coordinates (cx, cy) for a closed polygon defined by
    a sequence of (x, y) points. If the computed area is zero (e.g., colinear
    or insufficient points), returns the average of the input coordinates.

    Args:
        polygon (list[list[float]]): List of [x, y] pairs defining the polygon.

    Returns:
        tuple[float, float]: Centroid coordinates (cx, cy).
    """
    x = np.asarray([p[0] for p in polygon])
    y = np.asarray([p[1] for p in polygon])
    # Compute area using the shoelace method (1/2 sum(xi yi+1))
    area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    # If area = 0, that typically means that either the points are on a single line or
    # there are too few points to calculate area. If this is the case, we define the
    # centroid as the average of the x and y values.
    if not area:
        # Another edge case that **shouldn't** be a problem with JSON annotations is
        # if the length of the points is a single point. While this shouldn't happen
        # with closed points, we account for its possibility here.
        if len(polygon) < 2:
            return x[0], y[0]
        else:
            return np.mean(x[:-1]), np.mean(y[:-1])

    # Compute centroid x and y coordinates
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    # Note: Our inputs points don't have the constraint of being CCW, which means
    # it's possible (and likely) to get a negative value when it should be positive.
    # The absolute value is what we're after
    cx = np.abs((1 / (6 * area)) * np.sum((x + np.roll(x, -1)) * cross))
    cy = np.abs((1 / (6 * area)) * np.sum((y + np.roll(y, -1)) * cross))

    return cx, cy


def convert_to_microns(points: list[tuple], mpp_x: float, mpp_y: float) -> list[tuple]:
    """Convert annotation points from pixel units to micron units.

    This function scales a list of (x, y) or (x, y, z) coordinate tuples
    by the provided microns-per-pixel (MPP) values in the x and y
    directions. It is used to transform annotation geometry into real-world
    units for quantitative analysis.

    Args:
        points (list[tuple]): A list of (x, y) or (x, y, z) points in pixel units.
        mpp_x (float): Microns per pixel in the x-direction.
        mpp_y (float): Microns per pixel in the y-direction.

    Returns:
        list[tuple]: The same list of points scaled to micron units.
    """
    points_in_microns = [(p[0] * mpp_x, p[1] * mpp_y) for p in points]

    return points_in_microns


def create_histogram(
    x: list[float],
    title: str = None,
    xlab: str = None,
    ylab: str = None,
    vline: float = None,
    linelab: str = None,
) -> Figure:
    """Create a matplotlib histogram with an optional vertical reference line.

    Converts the input data to a NumPy array, builds a square (3"×3") figure,
    and plots a histogram with a sensible default bin count. If `vline` is
    provided, draws a dashed vertical line annotated by `linelab`.

    Args:
        x (list[float]): Data values to histogram.
        title (str, optional): Plot title. Defaults to None.
        xlab (str, optional): X-axis label. Defaults to None.
        ylab (str, optional): Y-axis label. Defaults to None.
        vline (float, optional): X-coordinate at which to draw a vertical line.
            Defaults to None.
        linelab (str, optional): Label for the vertical line in the legend.
            Defaults to None.

    Returns:
        matplotlib.figure.Figure: The figure containing the histogram.
    """
    # Convert to array and compute median
    x = np.array(x)

    # Set up a “pretty” style (you can try 'seaborn-darkgrid', 'ggplot', etc.)
    # plt.style.use("seaborn-darkgrid")

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(3, 3))

    # Histogram
    n, bins, patches = ax.hist(
        x,
        bins=min(30, max(len(x) // 3, 5)),
        color="#4C72B0",
        edgecolor="white",
        alpha=0.9,
    )
    if vline:
        ax.axvline(
            vline,
            color="#DD8452",  # contrasting line color
            linestyle="--",
            linewidth=2,
            label=f"{linelab} = {vline:.2f}",
        )

    # Titles and labels
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    if xlab:
        ax.set_xlabel(xlab, fontsize=12)
    if ylab:
        ax.set_ylabel(ylab, fontsize=12)

    # Legend
    ax.legend(fontsize=12, frameon=True)

    # Tweak tick params for readability
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Tight layout so labels don’t get cut off
    plt.tight_layout()

    return fig


def create_table(
    items: dict[str, Any],
    key_column: str = "Item",
    value_column: str = "Value",
    fontsize: int = 10,
    scale: tuple[float, float] = (0.75, 0.75),
    wrap_width: int = 25,
    col_widths: tuple[float, float] = (0.1, 0.1),
) -> plt.Figure:
    """Create a matplotlib Figure containing a two-column summary table.

    This function generates a simple key-value table from a dictionary,
    suitable for embedding into PDF reports. Text wrapping, font sizing,
    and cell dimensions are configurable.

    Args:
        items (dict[str, Any]): A mapping of labels to values for each row.
        key_column (str): Header text for the key/label column.
        value_column (str): Header text for the value column.
        fontsize (int): Font size used in the table.
        scale (tuple[float, float]): Scale factors for table width and height.
        wrap_width (int): Approximate character width at which to wrap cell text.
        col_widths (tuple[float, float]): Relative widths of the two columns.

    Returns:
        matplotlib.figure.Figure: A figure object containing the rendered table.
    """
    # Pre-wrap your text
    wrapped_keys = [textwrap.fill(str(k), wrap_width) for k in items]
    wrapped_vals = [textwrap.fill(str(v), wrap_width) for v in items.values()]

    cell_data = list(zip(wrapped_keys, wrapped_vals))
    col_labels = [key_column, value_column]

    # Make figure + hide axes
    plt.margins(x=0, y=0)
    fig, ax = plt.subplots()
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Create table with forced column widths
    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        colWidths=col_widths,
        bbox=[0, 0, 0.75, 0.75],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(*scale)

    # Bold only the header row
    for (row, _), cell in table.get_celld().items():
        text = cell.get_text()
        text.set_wrap(True)
        if row == 0:
            text.set_fontweight("bold")

    return fig


def draw_plot(
    pdf_canvas: Canvas,
    fig: plt.Figure,
    x: float,
    y: float,
) -> tuple[Canvas, float]:
    """Draw a Matplotlib figure onto a ReportLab PDF canvas.

    This function embeds a Matplotlib figure onto the given canvas by
    rendering it as a vector graphic (SVG). The figure is automatically
    centered horizontally on the page, and vertical placement is adjusted
    based on the figure height. If there is insufficient space on the
    current page, a new page is created.

    Args:
        pdf_canvas (Canvas): The ReportLab Canvas to draw on.
        fig (matplotlib.figure.Figure): The Matplotlib figure to render.
        x (float): The starting x-coordinate (overridden to center the image).
        y (float): The current y-coordinate to begin drawing from.

    Returns:
        tuple[Canvas, float]: The canvas and the updated y-coordinate after drawing.
    """
    from io import BytesIO

    from reportlab.graphics import renderPDF
    from svglib.svglib import svg2rlg

    # Adjust x to be centered
    dpi = fig.get_dpi()
    x = (letter[0] - fig.get_figwidth() * dpi) / 2 + fig.get_figwidth() * 8

    # Adjust y to be at the bottom of the image
    fig_height = fig.get_figheight() * dpi
    y = y - fig_height

    # Draw the figure as a vector image
    imgdata = BytesIO()
    fig.savefig(imgdata, format="svg")
    imgdata.seek(0)
    vector_drawing = svg2rlg(imgdata)

    # Before drawing, we need to check that there's room to draw. if not,
    # create a new page
    needed_space = fig_height / 1.5 + 14
    if y - needed_space < 50:
        pdf_canvas, y = draw_new_page(pdf_canvas)
        y = y - fig_height

    renderPDF.draw(vector_drawing, pdf_canvas, x=x, y=y)

    return pdf_canvas, y - 14 * 4  # Adjust the y value by height of four lines


def draw_text(
    pdf_canvas: Canvas,
    x: float,
    y: float,
    lines: list[str],
    centered: bool = False,
    line_height: float = 14.0,
    font_size: float = 12.0,
) -> float:
    """Draw multiple lines of text onto a ReportLab Canvas object.

    The first line is drawn in bold, and all subsequent lines use a
    standard font. Optionally centers each line of text relative to the
    page width. Returns the adjusted vertical position to allow for
    additional content below.

    Args:
        pdf_canvas (Canvas): The ReportLab Canvas to draw on.
        x (float): Starting x-coordinate for text placement.
        y (float): Starting y-coordinate for the first line of text.
        lines (list[str]): List of text lines to render.
        centered (bool): Whether to horizontally center the text. Defaults to False.
        line_height (float): Vertical space between lines in points.
        font_size (float): Font size for all text lines.

    Returns:
        float: The y-coordinate after all lines are rendered, offset for spacing.
    """
    for i, line in enumerate(lines):
        # We typically want the first line to be bolded
        if i == 0:
            pdf_canvas.setFont("Helvetica-Bold", font_size)
        else:
            pdf_canvas.setFont("Helvetica", font_size)

        # If the text needs to be centered, we need to adjust x accordingly
        x = (letter[0] - len(line) * (font_size / 2.25)) / 2 if centered else x
        pdf_canvas.drawString(x, y, line)
        y -= line_height

    return pdf_canvas, y - line_height * 2  # Decrease y to start new paragraph


def draw_table(
    pdf_canvas: Canvas,
    fig: plt.Figure,
    x: float,
    y: float,
    text: list[str] = "",
    font_size: float = 12.0,
) -> tuple[Canvas, float]:
    """Draw a rendered matplotlib table figure onto a PDF canvas.

    This function exports a matplotlib figure as a high-DPI PNG image,
    measures its true dimensions in points, and draws it centered onto the
    PDF canvas. If the image would overflow the current page, a new page
    is added. Optional header text is drawn above the image.

    Args:
        pdf_canvas (Canvas): The ReportLab Canvas to draw on.
        fig (matplotlib.figure.Figure): A matplotlib figure, typically containing a table.
        x (float): The starting x-coordinate for drawing (overridden for centering).
        y (float): The current y-coordinate to begin drawing from.
        text (list[str]): Optional header lines to draw above the table.
        font_size (float): Font size for the header text.

    Returns:
        tuple[Canvas, float]: The canvas and the updated y-coordinate after drawing.
    """
    from io import BytesIO

    from reportlab.lib.utils import ImageReader

    # Render to a tightly‑cropped PNG in memory
    dpi = fig.get_dpi()
    buf = BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
    )
    buf.seek(0)

    # Wrap it in an ImageReader so we can ask its true pixel size
    img = ImageReader(buf)
    pix_w, pix_h = img.getSize()  # pixels

    # Compute its size in ReportLab points (1 pt = 1/72 inch)
    #    since your PNG is written at `dpi` dots/inch:
    width_pt = pix_w * 72.0 / dpi
    height_pt = pix_h * 72.0 / dpi

    # If you want to center horizontally on a letter‑width page:
    page_w = letter[0]
    x = (page_w - width_pt / 2) / 2

    # Before drawing, we need to check that there's room to draw. if not,
    # create a new page
    needed_space = height_pt / 2 + len(text) * 14
    if y - needed_space < 50:
        pdf_canvas, y = draw_new_page(pdf_canvas)

    pdf_canvas.setFont("Helvetica-Bold", font_size)
    pdf_canvas, y = draw_text(pdf_canvas, x=50, y=y, lines=text)

    # Adjust y for best visual appearance
    # ReportLab’s `y` is the bottom of the image, so shift down by height
    y = y - height_pt / 2
    pdf_canvas.drawImage(
        img,
        x,
        y,
        width=width_pt / 1.5,
        height=height_pt / 1.5,
        mask="auto",
    )

    return pdf_canvas, y - 28


def draw_new_page(
    pdf_canvas: Canvas,
) -> tuple[Canvas, float]:
    """Add a new page to the PDF and reset the vertical position.

    This function finalizes the current page on the canvas and starts a
    new one. It resets the y-coordinate to a standard top margin for
    continued drawing.

    Args:
        pdf_canvas (Canvas): The ReportLab Canvas to add a new page to.

    Returns:
        tuple[Canvas, float]: The updated canvas and new starting y-coordinate.
    """
    pdf_canvas.showPage()
    y = letter[1] - 50
    return pdf_canvas, y


def fetch_annotations(gc: GirderClient, args: CLIArgumentParser) -> dict[str, Any]:
    """Fetch annotation data from Girder for a given image and expected labels.

    This function resolves the image item ID from a provided file ID,
    retrieves all annotations attached to that item, and then looks for
    annotations matching expected names. If multiple annotations exist
    with the same name, an error is raised. If an expected annotation is
    missing, it is recorded as None in the result.

    Args:
        gc (GirderClient): An authenticated Girder client.
        args (CLIArgumentParser): Parsed CLI arguments, including image ID
        and expected annotation filenames.

    Returns:
        dict[str, dict | None]: A dictionary mapping each expected annotation
        name to its full annotation data, or None if not found.
    """
    # The args.image_id is actually a file ID, not an item ID. The girder
    # client is expecting an item ID to retrieve annotations, which is what we
    # get here
    image_info = gc.get(f"file/{args.image_id}")
    item_id = image_info["itemId"]

    # We use the item ID to get metadata for all annotations. This will give
    # us the annotation ID we need to get the actual coordinates
    annotations = gc.get(
        "annotation",
        parameters=dict(itemId=item_id),
    )

    # Now, for each annotation name of interest, we want to identify
    # all annotations associated with that name
    annotation_names = [
        args.non_gsg_filename,
        args.gsg_filename,
        args.tubules_filename,
        args.arteries_filename,
        args.cortical_interstitium_filename,
        args.medullary_interstitium_filename,
    ]

    annotation_data: dict[str, dict] = {}
    for ann_name in annotation_names:
        # Girder allows multiple annotations of the same name to be attached to
        # a single image. We want to raise an exception if this occurs.
        matching_anns = [ann for ann in annotations if ann["annotation"]["name"] == ann_name]

        if len(matching_anns) > 1:
            raise ValueError(
                f"Multiple annotations named '{ann_name}' found on item " f"{item_id}. Please ensure names are unique."
            )

        # If there are no matching annotations, that's okay. We just need to
        # store this information so that the correct Banff scores are computed
        # with the data that's available
        if not matching_anns:
            annotation_data[ann_name] = None
            continue

        # With a single matching ID, we can safely retrieve the annotation data
        ann_id = matching_anns[0]["_id"]
        ann_coordinates = gc.get(f"annotation/{ann_id}")
        annotation_data[ann_name] = ann_coordinates

    return annotation_data


def fetch_mpp(gc: GirderClient, image_id: int, default_mpp: float = 0.25) -> tuple[float, float]:
    """Fetch microns-per-pixel (MPP) resolution from Girder metadata.

    This function retrieves the MPP values in the x and y directions from
    the tile metadata associated with a whole slide image. If no values are
    present, a default MPP value is used.

    Args:
        gc (GirderClient): An authenticated Girder client.
        image_id (int): The Girder file ID of the image.
        default_mpp (float): Default microns-per-pixel to use if not available.

    Returns:
        tuple[float, float]: The (x, y) microns-per-pixel values.
    """
    # We need to first get the tile information from the item ID
    image_info = gc.get(f"file/{image_id}")
    item_id = image_info["itemId"]
    tile_info = gc.get(f"item/{item_id}/tiles")

    # If mm per pixel is not available, a default is set for an mpp of 0.25
    mpp_x = tile_info.get("mm_x", default_mpp / 1000) * 1000
    mpp_y = tile_info.get("mm_y", default_mpp / 1000) * 1000
    return mpp_x, mpp_y


def get_boundaries(element: dict):
    """Gets the min and max for x and y values."""
    x = [p[0] for p in element["points"]]
    y = [p[1] for p in element["points"]]
    return min(x), max(x), min(y), max(y)


def k_nearest_polygons(cortex: list[dict[str, Any]], k: int = 6) -> list[dict[str, Any]]:
    """Find k-nearest neighbor polygons for each cortex section.

    Builds a KDTree of polygon centroids in each section, queries up to k
    closest neighbors for each polygon, computes the minimum positive edge
    distance to those neighbors, and stores it as 'nearest edge'.

    Args:
        cortex (list[dict[str, Any]]): List of cortex sections, each with a
            "structures" key containing polygons with "centroid" and "points".
        k (int): Number of nearest neighbors to query per polygon.

    Returns:
        list[dict[str, Any]]: The input cortex list with each structure dict
            augmented by a "nearest edge" key.
    """
    for ctx_section in cortex:
        # Retrieve all centroids from the structures within this section of cortex.
        # We also want the points for investigating the nearest edges of the nearest
        # structures
        centroids = [polygon["centroid"] for polygon in ctx_section["structures"]]
        combined_points = [polygon["points"] for polygon in ctx_section["structures"]]

        if not centroids:
            # We can't do anything if there are no structures in this cortex
            continue

        # Create a KDTree of all centroids (using default leaf_size=40)
        centroid_tree = KDTree(centroids)

        # We need to handle the case that there are fewer than k neighbors for this
        # section of cortex. If this is the case, we want to consider all structures
        # as the "nearest neighbors"
        n_available = len(centroids)
        k_eff = min(k, n_available)

        # Add KNN centroids to each polygon
        for polygon in ctx_section["structures"]:
            distance, index = centroid_tree.query([polygon["centroid"]], k=k_eff)
            # These next two lines help in the case that k_eff = 1
            distance = np.atleast_2d(distance)[0]
            index = np.atleast_2d(index)[0]
            nearest_structures = [combined_points[idx] for idx in index]

            nearest_edges = []
            for points in nearest_structures:
                edge = nearest_edge(polygon["points"], points)
                if edge > 0:
                    # We only want to consider edges that are greater than 0
                    # (i.e., not overlapping)
                    nearest_edges.append(edge)

            polygon["nearest edge"] = min(nearest_edges) if nearest_edges else 0

    return cortex


def nearest_edge(polygon_a: dict[str, Any], polygon_b: list[list[float]], k: int = 6) -> float:
    """Compute the minimum distance between two polygon boundaries.

    Builds KD-trees on each polygon's vertices, queries up to k nearest
    points to the opposing polygon's centroid, then returns the smallest
    Euclidean distance among all those neighbor pairs.

    Args:
        polygon_a (list[list[float]]): [[x, y], …] vertices of the first polygon.
        polygon_b (list[list[float]]): [[x, y], …] vertices of the second polygon.
        k (int): Max number of nearest vertices to consider on each polygon.

    Returns:
        float: Minimum edge distance between the two polygons.
    """
    # We will only be using 2D points, so we need to reshape our data
    tree_a = KDTree(np.array(polygon_a)[:, :2])
    tree_b = KDTree(np.array(polygon_b)[:, :2])

    # We want to first find the nearest vertices to our
    centroid_a = compute_polygon_centroid(polygon_a)
    centroid_b = compute_polygon_centroid(polygon_b)

    # Query tree A and tree B with the minimum of k and the size of the respective
    # polygon
    ka = min(k, len(polygon_a))
    kb = min(k, len(polygon_b))
    _, ia = tree_a.query([centroid_b], k=ka)
    _, ib = tree_b.query([centroid_a], k=kb)

    # Looping over both sets of nearest points is k^2 time, which is a constant
    nearest_a = [polygon_a[idx] for idx in ia[0]]
    nearest_b = [polygon_b[idx] for idx in ib[0]]
    edges: list[float] = [np.linalg.norm(np.array(pa) - np.array(pb)) for pa in nearest_a for pb in nearest_b]

    return min(edges)


def wilson_interval(k: int, n: int) -> tuple[float, float]:
    """Compute the 95% Wilson score confidence interval for a binomial proportion.

    This function estimates a confidence interval for a proportion using the
    Wilson score method, which is more accurate than the normal approximation
    when sample sizes are small or proportions are near 0 or 1.

    Args:
        k (int): Number of successes (e.g., sclerosed glomeruli).
        n (int): Total number of trials (e.g., total glomeruli evaluated).

    Returns:
        tuple[float, float]: The lower and upper bounds of the 95% confidence interval.

    Raises:
        ValueError: If n is zero.

    Reference:
        https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    """
    if n == 0:
        raise ValueError("Total number of trials (n) must be greater than zero.")

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


def within_boundaries(element: dict, box: tuple) -> bool:
    """Check if a polygon overlaps a rectangular boundary.

    Validates that `box` is (xmin, xmax, ymin, ymax), then returns True
    if any vertex in `element['points']` lies within or overlaps it.

    Args:
        element (dict): Annotation with 'points': list of [x, y] pairs.
        box (tuple): Boundaries as (xmin, xmax, ymin, ymax).

    Returns:
        bool: True if the polygon overlaps the box, else False.

    Raises:
        ValueError: If `box` does not have 4 items or is misordered.
    """
    # Validate box boundaries
    if len(box) != 4:
        raise ValueError("4 items required for boundaries: xmin, xmax, ymin, ymax")
    elif box[0] > box[1] or box[2] > box[3]:
        raise ValueError("Expected boundaries in order of xmin, xmax, ymin, ymax")

    # Extract x and y values of the element
    element_x = [p[0] for p in element["points"]]
    element_y = [p[1] for p in element["points"]]
    return (
        # How loose should we be with this restriction? As it stands, this is including
        # any polygon that overlaps at all with the cortical interstitium
        max(element_x) > box[0]
        and min(element_x) < box[1]
        and max(element_y) > box[2]
        and min(element_y) < box[3]
    )
