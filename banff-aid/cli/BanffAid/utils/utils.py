"""Utility functions."""

import math
import subprocess

import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw

# from girder_client import GirderClient


def convert_notebook_to_pdf(
    notebook_path: str, output_dir: str = None
) -> None:
    """Convert a Jupyter Notebook to a PDF using nbconvert."""
    command = ["jupyter", "nbconvert", "--to", "html", notebook_path]
    if output_dir:
        command.extend(["--output-dir", output_dir])
    # Run the command and check for errors
    subprocess.run(command, check=True)


def compute_max_distance(points):
    """Compute the maximum distance of points to an edge of a single polygon.

    Given a list of points defining a polygon,
    compute the maximum distance from an interior point to the edge.
    The distance is computed using cv2.distanceTransform with L2 norm.
    Returns the negative of the maximum distance.

    Args:
      points (list[list[float]]): A closed set of ordered points in x, y
      format.

    Returns (float):
      The maximum distance from an edge of a polygon.
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


def print_histogram(numbers_to_print: list[float]) -> None:
    """This takes a list of numbers and creates a histogram."""
    numbers_to_print.sort()
    sample_size = len(numbers_to_print)
    minimum_value = min(numbers_to_print)
    maximum_value = max(numbers_to_print)

    # Calculate Q1, Median, and Q3 (ish)
    middle_value = (maximum_value - minimum_value) / 2 + minimum_value

    # Offer four different bin-number options based on the sample size
    if sample_size < 10:
        n_bins = sample_size
    elif sample_size < 20:
        n_bins = 5
    elif sample_size < 30:
        n_bins = 10
    else:
        n_bins = 15

    # Calculate the binwidth
    bin_width = (maximum_value - (minimum_value)) / n_bins

    # Set the values for a single bin
    one_bin = "|##|"

    intervals = {}

    for i in range(n_bins):
        # Set interval bounds
        lower_bound = minimum_value + i * bin_width
        upper_bound = lower_bound + bin_width

        # Calculate number of values contained within the interval
        intervals[i] = sum(
            [
                val >= lower_bound and val < upper_bound
                for val in numbers_to_print
            ]
        )

        # Ensure the maximum value is included by adding 1 to the last interval
        intervals[i] += 1 if i + 1 == n_bins else 0

    # Identify the mode
    sample_mode = max(intervals.values())

    # Create the rows of the histogram
    all_rows = ""
    for i in range(sample_mode):
        row_string = ""
        for value in intervals.values():
            row_string += " " * 4 if sample_mode - i > value else one_bin
        all_rows += row_string + "\n"

    # Create the top and base of the histogram
    overbar = (
        "\n"
        + +len(intervals) * "===="
        + "\n"
        + (int(len(intervals) * 4 / 2 - min(24, int(len(intervals) * 4 / 2))))
        * " "
        + "HISTOGRAM OF TUBULE DIAMETERS\n"
        + +len(intervals) * "===="
        + "\n\n"
    )
    underbar = len(intervals) * "====" + "\n"
    axis_breaks = str(round(minimum_value, 3))
    for q in [str(round(middle_value, 3)), str(round(maximum_value, 3))]:
        alignment = len(intervals) * 4 // 2 - len(q) - 2
        print(alignment)
        axis_breaks += alignment * " " + q

    # Put it all together
    histogram = overbar + all_rows + underbar + axis_breaks + "\n"
    print(histogram)


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
