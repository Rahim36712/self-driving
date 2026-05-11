import cv2
import numpy as np
import time
from collections import deque

MODEL_PATH = "yolov8n.pt"
SEG_MODEL_PATH = "yolov8n-seg.pt"
CONFIDENCE_THRESHOLD = 0.4
YOLO_SKIP_FRAMES = 2
SEG_SKIP_FRAMES = 3

CLOSE_AREA_PCT = 0.08
NEAR_AREA_PCT = 0.03

LEFT_ZONE = 0.33
RIGHT_ZONE = 0.66

COLOR_CLOSE = (0, 0, 255)
COLOR_NEAR = (0, 165, 255)
COLOR_FAR = (0, 255, 0)

IMPORTANT_CLASSES = {0, 1, 2, 3, 5, 7, 9, 11}
SEG_COLORS = {
    0: (255, 100, 100), 1: (0, 200, 200), 2: (0, 180, 0),
    3: (0, 100, 255), 5: (255, 0, 100), 7: (100, 100, 255),
}


class YOLODetector:
    def __init__(self, model_path=MODEL_PATH):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            self.model.to('cuda')

        self.frame_count = 0
        self.last_detections = []
        self.last_yolo_frame = None
        self.fps_history = deque(maxlen=30)

    def run(self, frame):
        self.frame_count += 1

        if (self.frame_count % YOLO_SKIP_FRAMES != 0
                and self.last_detections is not None
                and self.last_yolo_frame is not None):
            return self.last_detections, self.last_yolo_frame

        t0 = time.time()
        results = self.model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)

        detections = []
        yolo_frame = frame.copy()

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    label = result.names.get(cls_id, f"cls_{cls_id}")
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    area = (x2 - x1) * (y2 - y1)

                    detections.append({
                        'label': label, 'confidence': conf,
                        'bbox': (x1, y1, x2, y2), 'center': (cx, cy),
                        'area': area, 'class_id': cls_id
                    })

        h, w = frame.shape[:2]
        frame_area = h * w
        for det in detections:
            det['distance'] = _classify_distance(det, frame_area)
            det['zone'] = _get_zone(det, w)

        yolo_frame = _draw_detections(yolo_frame, detections)
        self.fps_history.append(1.0 / max(time.time() - t0, 0.001))
        self.last_detections = detections
        self.last_yolo_frame = yolo_frame
        return detections, yolo_frame


class SegmentationEngine:
    def __init__(self, model_path=SEG_MODEL_PATH):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            import torch
            if torch.cuda.is_available():
                self.model.to('cuda')
            self.available = True
        except Exception:
            self.available = False

        self.frame_count = 0
        self.last_overlay = None
        self.last_masks_info = []

    def run(self, frame):
        if not self.available:
            return frame.copy(), []

        self.frame_count += 1
        if (self.frame_count % SEG_SKIP_FRAMES != 0
                and self.last_overlay is not None):
            return self.last_overlay, self.last_masks_info

        h, w = frame.shape[:2]
        overlay = frame.copy()
        masks_info = []

        try:
            results = self.model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
            if results and len(results) > 0:
                result = results[0]
                if result.masks is not None and result.boxes is not None:
                    for mask, box in zip(result.masks.data.cpu().numpy(), result.boxes):
                        cls_id = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        if cls_id not in IMPORTANT_CLASSES:
                            continue

                        label = result.names.get(cls_id, f"cls_{cls_id}")
                        color = SEG_COLORS.get(cls_id, (200, 200, 200))

                        mask_resized = cv2.resize(mask, (w, h))
                        binary = (mask_resized > 0.5).astype(np.uint8)
                        mask_area = np.sum(binary)
                        area_pct = mask_area / (w * h)

                        moments = cv2.moments(binary)
                        if moments["m00"] > 0:
                            cx = int(moments["m10"] / moments["m00"])
                            cy = int(moments["m01"] / moments["m00"])
                        else:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        zone = 'LEFT' if cx < w * LEFT_ZONE else ('RIGHT' if cx > w * RIGHT_ZONE else 'CENTER')
                        distance = 'CLOSE' if area_pct > CLOSE_AREA_PCT else ('NEAR' if area_pct > NEAR_AREA_PCT else 'FAR')

                        colored = np.zeros_like(overlay)
                        colored[binary == 1] = color
                        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.45, 0)
                        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay, contours, -1, color, 2)

                        tag = f"{label} {conf:.0%} {distance}"
                        cv2.putText(overlay, tag, (cx, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                        masks_info.append({
                            'label': label, 'class_id': cls_id, 'confidence': conf,
                            'center': (cx, cy), 'zone': zone, 'distance': distance,
                            'area_pct': area_pct
                        })
        except Exception:
            pass

        self.last_overlay = overlay
        self.last_masks_info = masks_info
        return overlay, masks_info

    def get_road_occupancy(self, masks_info):
        if not masks_info:
            return 0.0, None
        total = sum(m['area_pct'] for m in masks_info)
        center = [m for m in masks_info if m['zone'] == 'CENTER']
        primary = max(center, key=lambda m: m['area_pct']) if center else max(masks_info, key=lambda m: m['area_pct'])
        return min(total * 100, 100.0), primary


def _classify_distance(det, frame_area):
    pct = det['area'] / max(frame_area, 1)
    if pct > CLOSE_AREA_PCT:
        return 'CLOSE'
    elif pct > NEAR_AREA_PCT:
        return 'NEAR'
    return 'FAR'


def _get_zone(det, frame_width):
    cx = det['center'][0]
    if cx < frame_width * LEFT_ZONE:
        return 'LEFT'
    elif cx > frame_width * RIGHT_ZONE:
        return 'RIGHT'
    return 'CENTER'


def get_obstacle_threat(detections):
    if not detections:
        return 'CLEAR', []
    priority = {'CLOSE': 3, 'NEAR': 2, 'FAR': 1, 'CLEAR': 0}
    max_threat = 'CLEAR'
    blocked = []
    for det in detections:
        dist = det.get('distance', 'FAR')
        zone = det.get('zone', 'CENTER')
        if priority.get(dist, 0) > priority.get(max_threat, 0):
            max_threat = dist
        if dist in ('CLOSE', 'NEAR') and zone not in blocked:
            blocked.append(zone)
    return max_threat, blocked


def _draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        dist = det.get('distance', 'FAR')
        color = COLOR_CLOSE if dist == 'CLOSE' else (COLOR_NEAR if dist == 'NEAR' else COLOR_FAR)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det['label']} {det['confidence']:.0%} {dist}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"[{det.get('zone', '')}]", (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return frame


def draw_zone_lines(frame):
    h, w = frame.shape[:2]
    lx, rx = int(w * LEFT_ZONE), int(w * RIGHT_ZONE)
    cv2.line(frame, (lx, 0), (lx, h), (100, 100, 100), 1)
    cv2.line(frame, (rx, 0), (rx, h), (100, 100, 100), 1)
    cv2.putText(frame, "LEFT", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.putText(frame, "CENTER", (lx + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.putText(frame, "RIGHT", (rx + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    return frame


def detect_objects(frame, detector):
    """Run YOLO and return detections, threat level, blocked zones, annotated frame."""
    detections, yolo_frame = detector.run(frame)
    threat, blocked = get_obstacle_threat(detections)
    yolo_frame = draw_zone_lines(yolo_frame)
    return detections, threat, blocked, yolo_frame
