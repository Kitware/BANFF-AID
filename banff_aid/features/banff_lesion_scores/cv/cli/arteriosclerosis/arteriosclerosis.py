
from typing import Any

import numpy as np

def calculate_cv(
    intima: dict[str, dict], lumen: dict[str, dict], alpha: float = 1000.0
) -> int:
    """Calculate cv Banff lesion score.

    Args:
    intima (dict[str, dict]): Dictionary containing JSON-formatted annotations
                              for intimal wall (within artery).
    lumen (dict[str, dict]): Dictionary containing JSON-formatted annotations
                             for lumenal wall (within artery).
    alpha (float): Threshold for determining whether a lumen has a matching
                   intimal wall.

    Return value (int): cv score.
    """
    intima_polygons = [e["points"] for e in intima["annotation"]["elements"]]
    lumen_polygons = [e["points"] for e in lumen["annotation"]["elements"]]

    # Find the centers of each lumen and intima
    intima_centers = np.asarray(
        [
            (np.mean([x[0] for x in e]), np.mean([y[1] for y in e]))
            for e in intima_polygons
        ]
    )
    lumen_centers = np.asarray(
        [
            (np.mean([x[0] for x in e]), np.mean([y[1] for y in e]))
            for e in lumen_polygons
        ]
    )

    # Identify centers that are close to each other
    # - Note: If there is not intima associated with a given lumen, the artery
    #   is not affected by intimal thickening and is not of interest to us.
    shared_centers: list[tuple[float, float]] = []
    for i, int_center in enumerate(intima_centers):
        smallest_distance = alpha

        for j, lum_center in enumerate(lumen_centers):
            distance = np.sum((int_center - lum_center) ** 2)
            if distance < smallest_distance:
                # If new distance is smaller than the shortest distance, set
                # the new matching lumen to this lumen
                smallest_distance = distance
                lumen_match_index = j

        if smallest_distance < alpha:
            shared_centers.append((i, lumen_match_index))

    # Create a list of affected arteries with matching lumina and intimas.
    # We will calculate the proportion of the area contained by the lumen
    # and determine the most affected artery
    affected_arteries = [
        {
            "intima": np.asarray(intima_polygons[sc[0]]),
            "lumen": np.asarray(lumen_polygons[sc[1]]),
        }
        for sc in shared_centers
    ]

    # For each affected artery, we will create a loop around the boundary
    artery_luminal_reduction: list[dict[str, Any]] = []
    for artery in affected_arteries:
        intima = artery["intima"]
        lumen = artery["lumen"]

        # Calculate area for the intima and lumen
        intimal_area = polygon_area(intima)
        luminal_area = polygon_area(lumen)

        # Calculate the luminal reduction as the proportion of the area of the
        # inner-artery occupied by the lumen (i.e. 1 - luminal area proportion)
        # artery["luminal_reduction"] = 1 - (luminal_area) / (
        #     intimal_area + luminal_area
        # )
        artery["luminal_reduction"] = (
            intimal_area - luminal_area
        ) / intimal_area

        artery_luminal_reduction.append(artery)

    # Importantly, the cv score is based on the **worst** artery. This being
    # the case, we're going to find the maximum reduction
    max_reduction = max(
        artery_luminal_reduction,
        key=lambda artery: artery["luminal_reduction"],
    )["luminal_reduction"]

    # Using thresholds from 2022 Banff Foundation definition of cv score
    # - Ref: https://banfffoundation.org/central-repository-for-banff-
    #        classification-resources-3/
    cv_score = 0
    if 0.0 < max_reduction < 0.25:
        cv_score = 1
    elif 0.25 <= max_reduction < 0.5:
        cv_score = 2
    else:
        cv_score = 3
    print(
        f"\033[90mMaximum luminal reduction\033[0m: {round(max_reduction, 3)}"
    )
    print(f"\033[90mcv Score\033[0m: {cv_score}")

    return cv_score

def polygon_area(points: np.ndarray) -> float:
    """Calculate area of a polygon using the shoelace method.

    Args:
    points (np.ndarray): Nx2 or Nx3 array, with shape = (N, 2) or (N, 3).
    If Nx3, ignoring the z column if you only want area in XY plane.

    Return value (float): Area of the polygon.
    """
    # If you have [x, y, z] in each row, slice off the z:
    points_xy = points[:, :2]
    # Shoelace formula:
    x = points_xy[:, 0]
    y = points_xy[:, 1]

    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))