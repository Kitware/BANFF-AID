"""Utility functions."""

import math
import subprocess

import numpy as np
import scipy.ndimage as ndi
from girder_client import GirderClient
from PIL import Image, ImageDraw


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
    """Compute the maximum distance of points to an edge.

    Given a list of points defining a polygon,
    compute the maximum (signed) distance from an interior point to the edge.
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

    # Return negative value for signed convention (interior is negative)
    return -max_distance


def get_items_and_annotations(
    girder_client: GirderClient,
    folder_id: int,
    missing: bool = False,
):
    """Retrieve items from a folder along with their matching annotations.

    Args:
    - girder_client: An instance of GirderClient to communicate with the
      Girder server.
    - folder_id: The ID of the folder containing the items.
    - annotation_name: A string that should be part of the annotation's name.
    - missing: If False, return items with valid annotations. If True,
      return items missing valid annotations.

    Returns:
    - A list of tuples (item, annotation_record, first_element) if missing
      is False.
    - A list of items missing a valid annotation if missing is True.
    """
    results = []  # Initialize list to hold the results

    # Loop over all items in the specified folder
    for item in girder_client.listItem(folder_id):
        # Only process items that have a 'largeImage' property
        if not item.get("largeImage"):
            continue

        valid_annotation_found = False

        # Retrieve annotation records for the current item,
        # sorted by most recent update.
        annotation_records = girder_client.get(
            "annotation",
            parameters={
                "itemId": item["_id"],
                "sort": "updated",
                "sortdir": -1,
            },
        )

        # Process each annotation record for the item
        for annotation_record in annotation_records:
            # Get the annotation title for easier access
            annotation_title = annotation_record["annotation"]["name"]

            # Skip annotations that do not contain the desired name, or
            # that include 'Predictions'
            if "Predictions" in annotation_title:
                continue

            # Fetch full annotation details using the annotation record's
            # ID.
            annotation_detail = girder_client.get(
                f"annotation/{annotation_record['_id']}"
            )

            # Validate that the annotation structure contains 'elements'
            # and is non-empty.
            if (
                "annotation" not in annotation_detail
                or "elements" not in annotation_detail["annotation"]
                or not annotation_detail["annotation"]["elements"]
            ):
                continue

            # Get the first element from the annotation's elements list
            first_element = annotation_detail["annotation"]["elements"][0]

            # Check if the element is of type 'pixelmap' and contains a
            # bounding box.
            if first_element["type"] != "pixelmap" or not first_element.get(
                "user", {}
            ).get("bbox"):
                continue

            # If not in "missing" mode, append the valid item with
            # annotation details.
            if not missing:
                results.append((item, annotation_record, first_element))

            valid_annotation_found = True
            break  # Stop checking further annotations for this item.

        # If no valid annotation was found and we're checking for missing
        # annotations, add the item to the results list.
        if not valid_annotation_found and missing:
            results.append(item)

    return results


def major_minor_axes(points: list[list[float]]) -> tuple[float, float]:
    """Computes major and minor axes for an ellipse represented by x, y points.

    Args:
      tubules (dict[str, Any]):
        JSON annotations for tubules.

    Returns (tuple[float, float]):
      Length of major and minor axes of ellipse (format: major, minor)
    """
    if len(points) < 2:
        raise ValueError("At least two points are needed for an ellipse.")

    # Find the pair of points with the maximum Euclidean distance (the major
    # axis endpoints)
    max_distance = 0.0
    major_axis_pair = None
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            d = math.hypot(dx, dy)
            if d > max_distance:
                max_distance = d
                major_axis_pair = (p1, p2)

    major_axis = max_distance
    # Get the endpoints of the major axis
    (p1, p2) = major_axis_pair
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]

    # Compute the center of the ellipse (midpoint of the major axis)
    center = ((x1 + x2) / 2, (y1 + y2) / 2)

    # The line through (x1, y1) and (x2, y2) defines the major axis.
    # We'll compute the perpendicular distance from each point to this line.
    # The formula for distance from point (x0, y0) to the line through (x1, y1)
    # and (x2, y2):
    #   distance = |(y2 - y1)*x0 - (x2 - x1)*y0
    #              + (x2*y1 - y2*x1)| / sqrt((y2-y1)^2
    #              + (x2-x1)^2)
    # Reference: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    denom = math.hypot(x2 - x1, y2 - y1)
    max_perp_distance = 0.0
    for p in points:
        x0, y0 = p[0], p[1]
        dist = (
            abs((y2 - y1) * x0 - (x2 - x1) * y0 + (x2 * y1 - y2 * x1)) / denom
        )
        if dist > max_perp_distance:
            max_perp_distance = dist
            minor_axis_pair = (x0, y0)

    # The maximum perpendicular distance represents the semi-minor axis,
    # so the full minor axis length is twice that.
    minor_axis = 2 * max_perp_distance

    return major_axis, minor_axis
    # return {
    #     "major_axis": {"points": major_axis_pair, "distance": major_axis},
    #     "minor_axis": {"points": minor_axis_pair, "distance": minor_axis},
    #     "center": center
    # }


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
