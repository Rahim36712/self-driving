import cv2
import numpy as np

# --- Config ---
CANNY_LOW = 50
CANNY_HIGH = 150
BLUR_KERNEL = (5, 5)
HOUGH_RHO = 2
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LEN = 40
HOUGH_MAX_LINE_GAP = 150
ADAPTIVE_BLOCK = 11
ADAPTIVE_C = 2
LANE_COLOR = (0, 255, 0)
LANE_THICKNESS = 4
MIN_SLOPE = 0.3
POLY_DEGREE = 2
MIN_POLY_POINTS = 50


def apply_roi(frame, roi_ratio=0.6):
    # mask everything except bottom part of frame (where road is)
    h, w = frame.shape[:2]
    vertices = np.array([[
        (0, h),
        (int(w * 0.1), int(h * (1 - roi_ratio))),
        (int(w * 0.9), int(h * (1 - roi_ratio))),
        (w, h),
    ]], dtype=np.int32)
    mask = np.zeros_like(frame)
    fill = (255, 255, 255) if len(frame.shape) == 3 else 255
    cv2.fillPoly(mask, vertices, fill)
    return cv2.bitwise_and(frame, mask)


def canny_edges(frame):
    # standard pipeline: grayscale -> blur -> canny
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    return gray, edges


def adaptive_threshold_edges(frame):
    # alternative to canny, works better in uneven lighting
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK, ADAPTIVE_C
    )
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return gray, thresh


def _average_lines(lines):
    # average multiple line segments into one slope+intercept
    if not lines:
        return None
    slopes, intercepts = [], []
    for x1, y1, x2, y2, slope in lines:
        intercepts.append(y1 - slope * x1)
        slopes.append(slope)
    avg_slope = np.mean(slopes)
    avg_intercept = np.mean(intercepts)
    if abs(avg_slope) < 0.001:
        return None
    return avg_slope, avg_intercept


def _extrapolate_line(slope_intercept, y_bottom, y_top):
    if slope_intercept is None:
        return None
    slope, intercept = slope_intercept
    if abs(slope) < 0.001:
        return None
    x1 = int((y_bottom - intercept) / slope)
    x2 = int((y_top - intercept) / slope)
    return (x1, int(y_bottom), x2, int(y_top))


def detect_lanes_hough(edges):
    # hough transform to find straight lines, then separate left/right by slope
    lines = cv2.HoughLinesP(
        edges, rho=HOUGH_RHO, theta=HOUGH_THETA, threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN, maxLineGap=HOUGH_MAX_LINE_GAP
    )
    if lines is None:
        return None, None, None

    left_lines, right_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < MIN_SLOPE:
            continue
        if slope < 0:
            left_lines.append((x1, y1, x2, y2, slope))
        else:
            right_lines.append((x1, y1, x2, y2, slope))

    return _average_lines(left_lines), _average_lines(right_lines), lines


def fit_polynomial_lane(edge_image, side='left'):
    # fit a 2nd degree curve to edge pixels for curved roads
    h, w = edge_image.shape[:2]
    mid = w // 2
    if side == 'left':
        region = edge_image[:, :mid]
        x_offset = 0
    else:
        region = edge_image[:, mid:]
        x_offset = mid

    ys, xs = np.nonzero(region)
    if len(ys) < MIN_POLY_POINTS:
        return None, None

    xs = xs + x_offset
    try:
        coeffs = np.polyfit(ys, xs, POLY_DEGREE)
    except (np.linalg.LinAlgError, ValueError):
        return None, None

    y_points = np.linspace(int(h * 0.4), h - 1, num=100).astype(int)
    x_points = np.clip(np.polyval(coeffs, y_points).astype(int), 0, w - 1)
    return list(zip(x_points, y_points)), coeffs


def draw_curved_lanes(frame, left_pts, right_pts):
    overlay = frame.copy()
    if left_pts and len(left_pts) > 1:
        cv2.polylines(overlay, [np.array(left_pts, np.int32).reshape(-1, 1, 2)],
                      False, LANE_COLOR, LANE_THICKNESS)
    if right_pts and len(right_pts) > 1:
        cv2.polylines(overlay, [np.array(right_pts, np.int32).reshape(-1, 1, 2)],
                      False, LANE_COLOR, LANE_THICKNESS)
    if left_pts and right_pts and len(left_pts) > 1 and len(right_pts) > 1:
        polygon = np.vstack([np.array(left_pts, np.int32),
                             np.array(right_pts, np.int32)[::-1]]).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [polygon], (0, 255, 0))
        frame = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)
    else:
        frame = overlay
    return frame


def detect_lanes(frame, use_adaptive=False, use_poly=False):
    """
    Main lane detection function.
    Returns: left_line, right_line, offset, debug_edges, lane_frame, method
    """
    h, w = frame.shape[:2]

    if use_adaptive:
        gray, edges = adaptive_threshold_edges(frame)
        method = "Adaptive"
    else:
        gray, edges = canny_edges(frame)
        method = "Canny"

    roi_edges = apply_roi(edges)

    # try polynomial fitting if enabled
    left_pts, right_pts = None, None
    if use_poly:
        left_pts, _ = fit_polynomial_lane(roi_edges, 'left')
        right_pts, _ = fit_polynomial_lane(roi_edges, 'right')
        if left_pts or right_pts:
            method += "+Poly"

    # hough lines (always run for straight line detection)
    left_hough, right_hough, raw_lines = detect_lanes_hough(roi_edges)
    y_bottom = h - 1
    y_top = int(h * 0.4)
    left_line = _extrapolate_line(left_hough, y_bottom, y_top)
    right_line = _extrapolate_line(right_hough, y_bottom, y_top)
    if not (left_pts or right_pts):
        method += "+Hough"

    # draw on frame
    lane_frame = frame.copy()
    if use_poly and (left_pts or right_pts):
        lane_frame = draw_curved_lanes(lane_frame, left_pts, right_pts)
    if left_line is not None:
        cv2.line(lane_frame, (left_line[0], left_line[1]),
                 (left_line[2], left_line[3]), LANE_COLOR, LANE_THICKNESS)
    if right_line is not None:
        cv2.line(lane_frame, (right_line[0], right_line[1]),
                 (right_line[2], right_line[3]), LANE_COLOR, LANE_THICKNESS)

    # calculate how far lane center is from frame center
    offset = 0
    if left_line is not None and right_line is not None:
        lane_center = (left_line[0] + right_line[0]) // 2
        offset = lane_center - w // 2

    return left_line, right_line, offset, roi_edges, lane_frame, method
