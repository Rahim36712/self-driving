import cv2
import numpy as np

OFFSET_THRESHOLD = 80
ROAD_BLOCK_PCT = 25.0

DECISION_COLORS = {
    'FORWARD': (0, 220, 100), 'STOP': (0, 0, 255),
    'STOP - NO LANES': (0, 0, 255), 'STOP - BLOCKED': (0, 0, 255),
    'TURN LEFT': (0, 200, 255), 'TURN RIGHT': (0, 200, 255),
    'STEER LEFT': (0, 180, 220), 'STEER RIGHT': (0, 180, 220),
    'REVERSE': (180, 0, 255),
}


def make_driving_decision(left_line, right_line, offset, obstacle_threat,
                          blocked_zones, road_occupancy=0.0, seg_primary=None):
    """
    Rule-based decision engine. Checks conditions in priority order
    and returns (decision_string, reason_string).
    """
    # close obstacle -> immediate action
    if obstacle_threat == 'CLOSE':
        if 'CENTER' in blocked_zones:
            if len(blocked_zones) >= 3:
                return 'REVERSE', 'All zones blocked'
            return 'STOP', 'Obstacle blocking center'
        if 'LEFT' in blocked_zones and 'RIGHT' in blocked_zones:
            return 'STOP', 'Both sides blocked'
        if 'LEFT' in blocked_zones:
            return 'TURN RIGHT', 'Obstacle on left'
        if 'RIGHT' in blocked_zones:
            return 'TURN LEFT', 'Obstacle on right'

    if obstacle_threat == 'NEAR':
        if 'CENTER' in blocked_zones:
            return 'STOP', 'Obstacle approaching center'
        if 'LEFT' in blocked_zones:
            return 'TURN RIGHT', 'Obstacle nearby left'
        if 'RIGHT' in blocked_zones:
            return 'TURN LEFT', 'Obstacle nearby right'

    # segmentation says road is too blocked
    if road_occupancy > ROAD_BLOCK_PCT:
        lbl = seg_primary.get('label', 'obstacle') if seg_primary else 'obstacles'
        return 'STOP - BLOCKED', f'Road {road_occupancy:.0f}% blocked by {lbl}'

    # no lanes at all
    if left_line is None and right_line is None:
        return 'STOP - NO LANES', 'No lane markings detected'

    # one lane missing
    if left_line is None:
        return 'TURN LEFT', 'Left lane not detected'
    if right_line is None:
        return 'TURN RIGHT', 'Right lane not detected'

    # drifting too far from center
    if offset > OFFSET_THRESHOLD:
        return 'STEER LEFT', f'Drifting right ({offset:+d}px)'
    if offset < -OFFSET_THRESHOLD:
        return 'STEER RIGHT', f'Drifting left ({offset:+d}px)'

    return 'FORWARD', 'Clear path'


def draw_decision(frame, decision, reason="", fps=0.0):
    h, w = frame.shape[:2]
    color = DECISION_COLORS.get(decision, (200, 200, 200))

    # dark bar at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    # big decision text centered
    font = cv2.FONT_HERSHEY_SIMPLEX
    sz = cv2.getTextSize(decision, font, 1.2, 3)[0]
    tx = (w - sz[0]) // 2
    ty = h - 80 + sz[1] + 10
    cv2.putText(frame, decision, (tx + 2, ty + 2), font, 1.2, (0, 0, 0), 4)
    cv2.putText(frame, decision, (tx, ty), font, 1.2, color, 3)

    if reason:
        rsz = cv2.getTextSize(reason, font, 0.4, 1)[0]
        cv2.putText(frame, reason, ((w - rsz[0]) // 2, ty + 22), font, 0.4, (180, 180, 180), 1)

    if fps > 0:
        fcolor = (0, 255, 0) if fps >= 15 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), font, 0.6, fcolor, 2)

    return frame
