"""Vision for Bridge: what is the patient pointing at? Are they giving a thumbs-up?

Uses the Mac webcam, YOLO (finds objects like "cup", "tv", "remote") and
MediaPipe hand landmarks (finds the 21 points of the hand). Works offline:
both models are local files.

How the pointing target is chosen, each frame:
  "ray"     - draw a line from the index knuckle through the fingertip and keep
              going; the first object box it hits is the target.
  "nearest" - the line hits nothing: use the object whose center is closest to
              the fingertip (within NEAREST_MAX_PX pixels).
  "held"    - no hand visible: use the biggest object near the middle of the
              picture (the patient holds it up to the camera).
Then smoothing: an object only counts once it won most of the last 8 frames
(about 0.5 s), so a finger passing over something doesn't count.

Run it alone to test (opens a debug window, press q to quit):
    .venv/bin/python vision.py
    .venv/bin/python vision.py --camera 1      # a different camera

main.py uses it in the background: Vision().start(), then current_target().
"""
import os
os.environ.setdefault("YOLO_OFFLINE", "1")          # never let ultralytics go online
os.environ.setdefault("GLOG_minloglevel", "2")      # hide MediaPipe's INFO logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import math
import threading
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mpv

import ui

HERE = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL = os.path.join(HERE, "yolov8n.pt")
HAND_MODEL = os.path.join(HERE, "models", "hand_landmarker.task")

# ---------------- settings you can tune ----------------
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720   # ask the camera for this size
MAX_FPS = 16              # process at most this many frames per second (saves CPU)
YOLO_EVERY_N = 2          # run YOLO every 2nd frame; reuse its boxes in between
YOLO_MIN_CONF = 0.35      # ignore YOLO boxes it's less sure about than this
NEAREST_MAX_PX = 150      # "nearest" fallback: object center must be this close to the fingertip
HELD_CENTER_FRACTION = 0.5    # "held" fallback: object center must be in the middle 50%
SMOOTH_FRAMES = 8         # look at the last 8 frames (8 / 16 fps = 0.5 s)...
SMOOTH_MIN_VOTES = 5      # ...and the object must win at least 5 of them
TARGET_MAX_AGE_S = 2.0    # current_target() forgets a target older than this
THUMBS_UP_HOLD_S = 0.7    # thumbs-up must be held this long to count
THUMBS_UP_GRACE_S = 0.25  # a thumbs-up missing for less than this still counts as held

# MediaPipe hand landmark numbers
WRIST, THUMB_MCP, THUMB_IP, THUMB_TIP = 0, 2, 3, 4
INDEX_KNUCKLE, INDEX_TIP = 5, 8
MIDDLE_KNUCKLE = 9
FINGERS = [(6, 8), (10, 12), (14, 16), (18, 20)]   # (middle joint, tip) for the 4 fingers


# ---------------------------------------------------------------------------
# Geometry helpers (all in pixels)
# ---------------------------------------------------------------------------

def ray_hits_box(origin, direction, box):
    """How far along the ray (origin + t * direction) it first touches the box,
    or None if it misses. Standard "slab" test: check x and y ranges separately."""
    x1, y1, x2, y2 = box
    t_enter, t_exit = 0.0, float("inf")
    for o, d, lo, hi in ((origin[0], direction[0], x1, x2), (origin[1], direction[1], y1, y2)):
        if abs(d) < 1e-9:                 # ray runs parallel to these two sides
            if o < lo or o > hi:
                return None
            continue
        ta, tb = (lo - o) / d, (hi - o) / d
        t_enter = max(t_enter, min(ta, tb))
        t_exit = min(t_exit, max(ta, tb))
    return t_enter if t_enter <= t_exit else None


def box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def pick_pointed_object(knuckle, tip, objects):
    """objects: list of (label, box). Returns (label, method) or None."""
    dx, dy = tip[0] - knuckle[0], tip[1] - knuckle[1]
    length = math.hypot(dx, dy)
    if length > 1:
        direction = (dx / length, dy / length)
        hits = []
        for label, box in objects:
            t = ray_hits_box(tip, direction, box)     # the ray starts at the fingertip
            if t is not None:
                hits.append((t, label))
        if hits:
            return min(hits)[1], "ray"                # the first box along the ray

    # Nothing on the ray: closest object center to the fingertip, if close enough.
    best = None
    for label, box in objects:
        cx, cy = box_center(box)
        dist = math.hypot(cx - tip[0], cy - tip[1])
        if dist <= NEAREST_MAX_PX and (best is None or dist < best[0]):
            best = (dist, label)
    return (best[1], "nearest") if best else None


def pick_held_object(objects, width, height):
    """No hand visible: the biggest object whose center is in the middle of the frame."""
    margin_x = width * (1 - HELD_CENTER_FRACTION) / 2
    margin_y = height * (1 - HELD_CENTER_FRACTION) / 2
    central = [(box_area(box), label) for label, box in objects
               if margin_x <= box_center(box)[0] <= width - margin_x
               and margin_y <= box_center(box)[1] <= height - margin_y]
    return (max(central)[1], "held") if central else None


def is_thumbs_up(points):
    """points: 21 (x, y) hand landmarks in pixels (y grows DOWNWARD).
    Thumbs-up = thumb pointing up, clearly above all other fingertips,
    and the other 4 fingers curled into the palm."""
    hand_size = math.dist(points[WRIST], points[MIDDLE_KNUCKLE])
    if hand_size < 10:
        return False
    thumb_tip_y = points[THUMB_TIP][1]
    # Thumb points up: tip above its middle joint, which is above its base.
    if not (thumb_tip_y < points[THUMB_IP][1] < points[THUMB_MCP][1]):
        return False
    # Thumb tip clearly above every other fingertip (by 40% of the hand size).
    if thumb_tip_y > min(points[tip][1] for _, tip in FINGERS) - 0.4 * hand_size:
        return False
    # Each finger is curled: its tip is closer to the wrist than its middle joint.
    wrist = points[WRIST]
    for joint, tip in FINGERS:
        if math.dist(points[tip], wrist) > math.dist(points[joint], wrist):
            return False
    return True


# ---------------------------------------------------------------------------
# The Vision class
# ---------------------------------------------------------------------------

class Vision:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.lock = threading.Lock()       # protects the values main.py reads
        self.running = False
        self.thread = None
        self.web = False                   # True: _loop keeps a JPEG of each annotated frame
        self.jpeg = None                   # latest annotated frame as JPEG bytes (web mode)
        self.jpeg_no = 0                   # goes up by 1 with each new JPEG

        # Shared results (read by main.py from another thread)
        self._target = None                # {"label", "method", "time"}
        self.thumbs_up_count = 0           # goes up by 1 each time a thumbs-up is held 0.7 s

        # Per-frame working state
        self.frame_no = 0
        self.objects = []                  # [(label, box)] from the last YOLO run
        self.hand = None                   # 21 (x, y) points, or None
        self.votes = deque(maxlen=SMOOTH_FRAMES)   # recent (label, method) or None
        self.thumbs_since = None           # when the current thumbs-up started
        self.thumbs_last_seen = 0.0
        self.thumbs_fired = False          # already counted this thumbs-up?
        self.fps = 0.0
        self._fps_frames, self._fps_start = 0, time.time()

    # ---------- setup ----------
    def open(self):
        """Open the camera and load both models. Returns (ok, message)."""
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)  # macOS backend
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not self.cap.isOpened():
            return False, (f"Could not open camera {self.camera_index}. Check System Settings > "
                           "Privacy & Security > Camera for your terminal app.")
        # Read for ~1.5 s: the first frames can be missing or dark while the camera adjusts.
        frame, started = None, time.time()
        while time.time() - started < 1.5:
            ok, picture = self.cap.read()
            if ok:
                frame = picture
        if frame is None:
            return False, "The camera opened but sent no pictures (permission, or in use?)."
        self.warning = None
        if frame.mean() < 20:              # 0 = black, 255 = white
            self.warning = (f"Camera {self.camera_index} looks almost black (lid closed or lens "
                            "covered?). Try --camera 1.")

        from ultralytics import YOLO       # imported here because it is slow to load
        self.yolo = YOLO(YOLO_MODEL)
        self.device = "cpu"
        try:                               # use the Mac GPU ("mps") if it works
            self.yolo.predict(np.zeros((64, 64, 3), np.uint8), device="mps", verbose=False)
            self.device = "mps"
        except Exception:
            pass
        self.hands = mpv.HandLandmarker.create_from_options(mpv.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=mpv.RunningMode.VIDEO,    # VIDEO = tracks the hand between frames
            num_hands=1,
        ))
        self.start_time = time.time()
        message = f"camera {self.camera_index} open, YOLO on {self.device}"
        if self.warning:
            message += "\nWARNING: " + self.warning
        return True, message

    # ---------- one frame ----------
    def step(self, draw=False):
        """Read and process one frame. Returns the debug picture if draw=True."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)                  # mirror, like looking in a mirror
        height, width = frame.shape[:2]
        now = time.time()
        self.frame_no += 1
        self._fps_frames += 1                       # FPS, measured over each second
        if now - self._fps_start >= 1.0:
            self.fps = self._fps_frames / (now - self._fps_start)
            self._fps_frames, self._fps_start = 0, now

        # 1) YOLO objects (every Nth frame), ignoring people
        if self.frame_no % YOLO_EVERY_N == 1 or YOLO_EVERY_N == 1:
            result = self.yolo.predict(frame, device=self.device, conf=YOLO_MIN_CONF,
                                       verbose=False)[0]
            self.objects = []
            for box, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
                label = result.names[int(cls)]
                if label != "person":
                    self.objects.append((label, box))

        # 2) Hand landmarks (every frame). MediaPipe wants RGB and a rising timestamp.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        timestamp_ms = int((now - self.start_time) * 1000) + self.frame_no  # always increases
        found = self.hands.detect_for_video(image, timestamp_ms)
        self.hand = None
        if found.hand_landmarks:
            self.hand = [(p.x * width, p.y * height) for p in found.hand_landmarks[0]]

        # 3) This frame's guess: ray / nearest if a hand is visible, else held
        if self.hand:
            guess = pick_pointed_object(self.hand[INDEX_KNUCKLE], self.hand[INDEX_TIP], self.objects)
        else:
            guess = pick_held_object(self.objects, width, height)

        # 4) Smoothing: the label must win most of the last few frames
        self.votes.append(guess)
        labels = Counter(g[0] for g in self.votes if g)
        if labels:
            label, count = labels.most_common(1)[0]
            if count >= SMOOTH_MIN_VOTES:
                method = next(g[1] for g in reversed(self.votes) if g and g[0] == label)
                with self.lock:
                    self._target = {"label": label, "method": method, "time": now}

        # 5) Thumbs-up, held for THUMBS_UP_HOLD_S
        self._update_thumbs_up(self.hand is not None and is_thumbs_up(self.hand), now)

        if draw:
            return self._draw(frame)
        return frame

    def _update_thumbs_up(self, seen, now):
        if seen:
            if self.thumbs_since is None:
                self.thumbs_since = now
            self.thumbs_last_seen = now
            if not self.thumbs_fired and now - self.thumbs_since >= THUMBS_UP_HOLD_S:
                self.thumbs_fired = True
                with self.lock:
                    self.thumbs_up_count += 1
        elif self.thumbs_since is not None and now - self.thumbs_last_seen > THUMBS_UP_GRACE_S:
            self.thumbs_since = None           # thumb went away: reset for the next one
            self.thumbs_fired = False

    # ---------- what main.py reads ----------
    def current_target(self):
        """{"label", "method", "time"} of what the patient points at, or None if
        there's been no target for TARGET_MAX_AGE_S seconds."""
        with self.lock:
            t = self._target
        if t is None or time.time() - t["time"] > TARGET_MAX_AGE_S:
            return None
        return dict(t)

    def get_thumbs_up_progress(self):
        """(0-1 progress, counted?) of the thumbs-up being held now, or None."""
        since = self.thumbs_since
        if since is None:
            return None
        return min(1.0, (time.time() - since) / THUMBS_UP_HOLD_S), self.thumbs_fired

    def get_thumbs_up_count(self):
        with self.lock:
            return self.thumbs_up_count

    def set_pick_overlay(self, options, highlight):
        """Show these options in the window's panel, `highlight` (0-based) in yellow.
        Pass options=None to stop highlighting. Safe to call from another thread."""
        if options:
            ui.state.update(options=list(options), highlight=highlight)
        else:
            ui.state.update(highlight=None)

    # ---------- background thread (used by main.py) ----------
    def start(self):
        """Open the camera and start processing in the background. Returns (ok, message)."""
        ok, message = self.open()
        if not ok:
            return False, message
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True, message

    def _loop(self):
        while self.running:
            started = time.time()
            picture = self.step(draw=self.web)
            if self.web and picture is not None:
                ok, jpeg = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self.lock:
                        self.jpeg, self.jpeg_no = jpeg.tobytes(), self.jpeg_no + 1
            # Wait so we run at about MAX_FPS (the camera itself sends ~30).
            time.sleep(max(0.0, 1 / MAX_FPS - (time.time() - started)))

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()

    # ---------- drawing ----------
    def _draw(self, frame):
        """The whole app window (camera + side panel); the look lives in ui.py."""
        thumbs = self.get_thumbs_up_progress()
        knuckle = self.hand[INDEX_KNUCKLE] if self.hand else None
        tip = self.hand[INDEX_TIP] if self.hand else None
        if self.web:                       # the page draws the panel; send just the camera
            return ui.render_camera(frame, self.objects, self.hand, self.current_target(),
                                    knuckle, tip)
        return ui.render(frame, self.objects, self.hand, self.current_target(), thumbs,
                         knuckle, tip)


# ---------------------------------------------------------------------------
# Run alone: debug window
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bridge vision debug window")
    parser.add_argument("--camera", type=int, default=0, help="camera number (default 0)")
    args = parser.parse_args()

    vision = Vision(args.camera)
    ok, message = vision.open()
    print(message)
    if not ok:
        return
    print("Point at things (cup, bottle, remote, tv, book...). Thumbs-up to test. q = quit.")
    last_print = time.time()
    last_target, last_thumbs = None, 0
    while True:
        started = time.time()
        picture = vision.step(draw=True)
        if picture is None:
            print("camera stopped sending frames")
            break
        cv2.imshow("Bridge vision (q = quit)", picture)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if time.time() - last_print >= 2.0:
            print(f"FPS {vision.fps:.1f}")
            last_print = time.time()

        target = vision.current_target()
        name = f"{target['label']} ({target['method']})" if target else None
        if name != last_target:
            print(f"pointing at: {name or 'nothing'}")
            last_target = name
        if vision.get_thumbs_up_count() != last_thumbs:
            last_thumbs = vision.get_thumbs_up_count()
            print("THUMBS UP")
        time.sleep(max(0.0, 1 / MAX_FPS - (time.time() - started)))
    vision.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
