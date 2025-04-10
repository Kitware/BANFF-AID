"""Sandbox."""

import json

import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw


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


# Load the JSON file (adjust your path as needed)
with open(
    "../../../../CMIL/train/convert-xml-json/example/542-converted.json"
) as file:
    data = json.load(file)

elements = data["annotation"]["elements"]

# # Optionally scale down the annotations if needed
# scale_factor = 50
# for i, element in enumerate(elements):
#     points = element["points"]
#     points_scaled_down = [
#         [p[0] / scale_factor, p[1] / scale_factor] for p in points
#     ]
#     elements[i]["points"] = points_scaled_down

# For each element, compute the maximum distance from the edge
for element in elements:
    element["max_distance"] = compute_max_distance(element["points"])

# Print the maximum distance for each element
for i, element in enumerate(elements):
    print(f"Element {i} max_distance: {element['max_distance']}")
