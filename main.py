import cv2
import numpy as np
import time
import glob
import os
import sys
from datetime import datetime

from lane_detection import detect_lanes
from object_detection import YOLODetector, SegmentationEngine, detect_objects, draw_zone_lines
from decision import make_driving_decision, draw_decision

# --- Config ---
FRAME_W = 960
FRAME_H = 540
MULTI_W = 480
MULTI_H = 270


class VideoStream:
    """Processes one video stream with lane detection + YOLO + segmentation."""

    def __init__(self, source, detector, seg_engine=None, stream_id=0, is_webcam=False):
        self.detector = detector
        self.seg_engine = seg_engine
        self.stream_id = stream_id
        self.is_webcam = is_webcam

        if isinstance(source, cv2.VideoCapture):
            self.cap = source
            self.video_name = "Webcam (Live)"
        else:
            self.cap = cv2.VideoCapture(source)
            self.video_name = os.path.basename(source)

        self.current_frame = None
        self.frame_count = 0
        self.lane_result = None
        self.yolo_result = None
        self.seg_result = None
        self.road_occ = 0.0
        self.seg_primary = None
        self.decision = "STARTING..."
        self.reason = ""
        self.fps = 0.0
        self.use_adaptive = False
        self.use_poly = False
        self.use_seg = False
        self.fps_history = []

    def process_frame(self, target_size=None):
        ret, raw = self.cap.read()
        if not ret:
            if self.is_webcam:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, raw = self.cap.read()
            if not ret:
                return None

        w = target_size[0] if target_size else FRAME_W
        h = target_size[1] if target_size else FRAME_H
        frame = cv2.resize(raw, (w, h))
        self.current_frame = frame.copy()
        self.frame_count += 1
        t0 = time.time()

        # lane detection (classical)
        left, right, offset, edges, lane_frame, method = detect_lanes(
            frame, self.use_adaptive, self.use_poly)
        self.lane_result = {
            'left': left, 'right': right, 'offset': offset,
            'edges': edges, 'lane_frame': lane_frame, 'method': method
        }

        # yolo detection (deep learning)
        dets, threat, blocked, yolo_frame = detect_objects(frame, self.detector)
        self.yolo_result = {
            'detections': dets, 'threat': threat,
            'blocked': blocked, 'yolo_frame': yolo_frame
        }

        # segmentation (if enabled)
        self.road_occ = 0.0
        self.seg_primary = None
        if self.use_seg and self.seg_engine and self.seg_engine.available:
            seg_overlay, masks = self.seg_engine.run(frame)
            self.seg_result = {'overlay': seg_overlay, 'masks': masks}
            self.road_occ, self.seg_primary = self.seg_engine.get_road_occupancy(masks)
        else:
            self.seg_result = None

        # decision
        self.decision, self.reason = make_driving_decision(
            left, right, offset, threat, blocked, self.road_occ, self.seg_primary)

        elapsed = time.time() - t0
        self.fps = 1.0 / max(elapsed, 0.001)
        self.fps_history.append(self.fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        return frame

    def avg_fps(self):
        return sum(self.fps_history) / max(len(self.fps_history), 1)

    def build_output(self, frame):
        if frame is None:
            return np.zeros((FRAME_H, FRAME_W, 3), np.uint8)

        out = frame.copy()
        h, w = out.shape[:2]

        # draw lane area + lines
        if self.lane_result:
            left = self.lane_result['left']
            right = self.lane_result['right']
            if left and right:
                pts = np.array([[left[0], left[1]], [left[2], left[3]],
                                [right[2], right[3]], [right[0], right[1]]], np.int32)
                ov = out.copy()
                cv2.fillPoly(ov, [pts], (0, 180, 0))
                out = cv2.addWeighted(out, 0.75, ov, 0.25, 0)
            if left:
                cv2.line(out, (left[0], left[1]), (left[2], left[3]), (0, 255, 0), 3)
            if right:
                cv2.line(out, (right[0], right[1]), (right[2], right[3]), (0, 255, 0), 3)

        # draw yolo boxes
        if self.yolo_result:
            for det in self.yolo_result['detections']:
                x1, y1, x2, y2 = det['bbox']
                dist = det.get('distance', 'FAR')
                color = (0, 0, 255) if dist == 'CLOSE' else ((0, 165, 255) if dist == 'NEAR' else (0, 220, 0))
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                tag = f"{det['label']} {det['confidence']:.0%} {dist}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
                cv2.putText(out, tag, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            out = draw_zone_lines(out)

        out = draw_decision(out, self.decision, self.reason, self.avg_fps())

        # status bar
        lr = self.lane_result or {}
        yr = self.yolo_result or {}
        info = f"L:{'Y' if lr.get('left') else 'N'} R:{'Y' if lr.get('right') else 'N'} | " \
               f"Objs:{len(yr.get('detections', []))} Threat:{yr.get('threat', 'N/A')} | {lr.get('method', '')}"
        cv2.putText(out, info, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        return out

    def build_debug(self, frame):
        h, w = frame.shape[:2] if frame is not None else (FRAME_H, FRAME_W)
        pw, ph = w // 3, h // 2

        def rp(img):
            if img is None:
                return np.zeros((ph, pw, 3), np.uint8)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return cv2.resize(img, (pw, ph))

        def lbl(panel, text):
            cv2.putText(panel, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)
            return panel

        lr = self.lane_result or {}
        yr = self.yolo_result or {}

        p1 = lbl(rp(frame), "ORIGINAL")
        p2 = lbl(rp(lr.get('edges')), f"EDGES [{lr.get('method', '')}]")
        p3 = lbl(rp(lr.get('lane_frame')),
                 f"LANES L:{'Y' if lr.get('left') else 'N'} R:{'Y' if lr.get('right') else 'N'}")
        p4 = lbl(rp(yr.get('yolo_frame')), f"YOLO [{yr.get('threat', 'N/A')}]")

        # panel 5: segmentation or merged view
        if self.seg_result and self.seg_result.get('overlay') is not None:
            p5 = lbl(rp(self.seg_result['overlay']), f"SEGMENTATION ({len(self.seg_result.get('masks', []))} obj)")
        else:
            p5 = lbl(rp(self.build_output(frame)), "MERGED")

        # decision panel
        p6 = np.zeros((ph, pw, 3), np.uint8)
        p6[:] = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        dec_colors = {
            'FORWARD': (0, 220, 100), 'STOP': (0, 0, 255), 'STOP - NO LANES': (0, 0, 255),
            'STOP - BLOCKED': (0, 0, 255), 'TURN LEFT': (0, 200, 255), 'TURN RIGHT': (0, 200, 255),
            'STEER LEFT': (0, 180, 220), 'STEER RIGHT': (0, 180, 220), 'REVERSE': (180, 0, 255),
        }
        c = dec_colors.get(self.decision, (200, 200, 200))
        cv2.putText(p6, "DECISION ENGINE", (10, 25), font, 0.5, (100, 100, 100), 1)
        fps = self.avg_fps()
        fc = (0, 255, 0) if fps >= 15 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
        cv2.putText(p6, f"FPS: {fps:.1f}", (pw - 110, 25), font, 0.5, fc, 1)
        sz = cv2.getTextSize(self.decision, font, 0.9, 2)[0]
        cv2.putText(p6, self.decision, ((pw - sz[0]) // 2, ph // 2), font, 0.9, c, 2)
        if self.reason:
            rsz = cv2.getTextSize(self.reason, font, 0.35, 1)[0]
            cv2.putText(p6, self.reason, ((pw - rsz[0]) // 2, ph // 2 + 25), font, 0.35, (150, 150, 150), 1)

        top = np.hstack([p1, p2, p3])
        bot = np.hstack([p4, p5, p6])
        return np.vstack([top, bot])

    def release(self):
        if self.cap:
            self.cap.release()


class MultiStream:
    def __init__(self, video_paths, detector, seg_engine=None, max_streams=4):
        n = min(len(video_paths), max_streams)
        self.streams = [VideoStream(video_paths[i], detector, seg_engine, i) for i in range(n)]

    def process(self):
        frames = []
        for s in self.streams:
            frame = s.process_frame(target_size=(MULTI_W, MULTI_H))
            if frame is not None:
                out = s.build_output(frame)
                cv2.putText(out, f"Stream {s.stream_id+1}: {s.video_name[:25]}",
                            (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)
                frames.append(out)
            else:
                frames.append(np.zeros((MULTI_H, MULTI_W, 3), np.uint8))
        while len(frames) < 4:
            frames.append(np.zeros((MULTI_H, MULTI_W, 3), np.uint8))
        return np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:4])])

    def release(self):
        for s in self.streams:
            s.release()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")
    ss_dir = os.path.join(script_dir, "screenshots")
    os.makedirs(ss_dir, exist_ok=True)

    videos = []
    for ext in ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.ts"]:
        videos.extend(glob.glob(os.path.join(dataset_dir, ext)))
    videos.sort()

    if not videos:
        print(f"No videos found in {dataset_dir}")
        sys.exit(1)

    print(f"Found {len(videos)} video(s)")
    for i, v in enumerate(videos):
        print(f"  [{i+1}] {os.path.basename(v)}")

    detector = YOLODetector()
    seg_engine = SegmentationEngine()

    idx = 0
    multi_mode = False
    debug_mode = False
    paused = False
    webcam = False

    stream = VideoStream(videos[idx], detector, seg_engine)
    multi = None

    print(f"\nPlaying: {stream.video_name}")
    print("Keys: Q=quit D=debug P=pause A=adaptive F=polyfit G=seg N=next M=multi W=webcam S=screenshot\n")

    while True:
        if paused:
            k = cv2.waitKey(100) & 0xFF
            if k == ord('p'):
                paused = False
            elif k == ord('q'):
                break
            continue

        if multi_mode:
            if multi is None:
                cv2.destroyAllWindows()
                multi = MultiStream(videos, detector, seg_engine)
            cv2.imshow("Multi-Stream", multi.process())
        else:
            if multi:
                multi.release()
                multi = None
                cv2.destroyAllWindows()

            frame = stream.process_frame()
            if frame is None:
                continue

            if debug_mode:
                cv2.imshow(f"Debug - {stream.video_name}", stream.build_debug(frame))
            else:
                cv2.imshow(f"Self-Driving - {stream.video_name}", stream.build_output(frame))

            if stream.frame_count % 20 == 0:
                lr = stream.lane_result or {}
                yr = stream.yolo_result or {}
                print(f"FPS:{stream.avg_fps():5.1f} | "
                      f"L:{'Y' if lr.get('left') else 'N'} R:{'Y' if lr.get('right') else 'N'} | "
                      f"Threat:{yr.get('threat','?'):5s} | >> {stream.decision}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            cv2.destroyAllWindows()
        elif key == ord('p'):
            paused = True
        elif key == ord('a') and not multi_mode:
            stream.use_adaptive = not stream.use_adaptive
        elif key == ord('f') and not multi_mode:
            stream.use_poly = not stream.use_poly
        elif key == ord('g') and not multi_mode:
            stream.use_seg = not stream.use_seg
        elif key == ord('w'):
            cv2.destroyAllWindows()
            webcam = not webcam
            stream.release()
            if webcam:
                cam = cv2.VideoCapture(0)
                if cam.isOpened():
                    stream = VideoStream(cam, detector, seg_engine, is_webcam=True)
                else:
                    webcam = False
                    stream = VideoStream(videos[idx], detector, seg_engine)
            else:
                stream = VideoStream(videos[idx], detector, seg_engine)
        elif key == ord('n') and not multi_mode and not webcam:
            cv2.destroyAllWindows()
            stream.release()
            idx = (idx + 1) % len(videos)
            stream = VideoStream(videos[idx], detector, seg_engine)
        elif key == ord('m'):
            multi_mode = not multi_mode
            webcam = False
            cv2.destroyAllWindows()
            if not multi_mode:
                stream = VideoStream(videos[idx], detector, seg_engine)
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(ss_dir, f"ss_{ts}.png")
            if multi_mode and multi:
                cv2.imwrite(path, multi.process())
            elif stream.current_frame is not None:
                img = stream.build_debug(stream.current_frame) if debug_mode else stream.build_output(stream.current_frame)
                cv2.imwrite(path, img)

    stream.release()
    if multi:
        multi.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
