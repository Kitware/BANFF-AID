import math

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
        dist = abs((y2 - y1)*x0 - (x2 - x1)*y0 + (x2*y1 - y2*x1)) / denom
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
            [val >= lower_bound and val < upper_bound for val in numbers_to_print]
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
        + (int(len(intervals) * 4 / 2 - min(24, int(len(intervals) * 4 / 2 )))) * " "
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
