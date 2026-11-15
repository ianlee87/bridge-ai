"""The camera window's look: camera on the left, a dark panel on the right.

Only drawing lives here. main.py tells it what to show with `state.update(...)`
(status, what was heard, the options, ...); vision.py calls `render(...)` each
frame. Text is drawn with PIL so it can use a real font (Helvetica Neue).
"""
import threading

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 720
MARGIN = 20
PANEL_W = 428                                   # about 35% of the window
CAM = (MARGIN, MARGIN, WIDTH - MARGIN * 2 - PANEL_W, HEIGHT - MARGIN)   # x1, y1, x2, y2
PANEL = (CAM[2] + MARGIN, MARGIN, WIDTH - MARGIN, HEIGHT - MARGIN)
PAD = 28                                        # padding inside the panel
RADIUS = 18

# Colors (RGB)
BG = (16, 16, 19)
PANEL_BG = (28, 28, 32)
CARD_BG = (42, 42, 48)
TEXT = (240, 240, 242)
MUTED = (140, 140, 150)
FAINT = (95, 95, 104)
DARK_TEXT = (24, 24, 26)
YELLOW = (250, 222, 122)
GREEN = (98, 200, 130)
RED = (214, 52, 52)
BLUE = (110, 160, 250)
STATUS_COLORS = {"idle": MUTED, "listening": (240, 90, 90), "thinking": BLUE,
                 "picking": YELLOW, "speaking": GREEN}


# ---------------------------------------------------------------------------
# What the panel shows (set by main.py, read by the drawing code)
# ---------------------------------------------------------------------------

class UIState:
    def __init__(self):
        self._lock = threading.Lock()
        self._values = {
            "status": "starting",  # starting / idle / listening / thinking / picking / speaking
            "heard": None,         # "drink... please"
            "fragments": [],       # ["drink.", "please."]
            "caregiver": None,     # the caregiver's question, until it is answered
            "pointing": None,      # what the patient pointed at for this turn
            "pointing_method": None,    # ray / nearest / held
            "pointing_locked": False,   # False = show the live camera target instead
            "options": [],
            "highlight": None,     # index of the yellow card (thumbs-up mode)
            "chosen": None,        # index of the green card (spoken)
            "urgent": None,        # "chest pain"
            "timings": {},         # {"stt": s, "bonsai": s, "tts": s}
        }

    def update(self, **values):
        with self._lock:
            if "timings" in values:          # merge, so stt/bonsai/tts can arrive separately
                values["timings"] = {**self._values["timings"], **values["timings"]}
            self._values.update(values)

    def new_turn(self):
        """Clear the last turn's results (called when the patient starts speaking)."""
        self.update(heard=None, fragments=[], pointing=None, pointing_method=None,
                    pointing_locked=False, options=[],
                    highlight=None, chosen=None, urgent=None)
        with self._lock:
            self._values["timings"] = {}

    def snapshot(self):
        with self._lock:
            return dict(self._values)


state = UIState()


# ---------------------------------------------------------------------------
# Fonts and text helpers
# ---------------------------------------------------------------------------

_FONT_FILE = "/System/Library/Fonts/HelveticaNeue.ttc"
_FACES = {"regular": 0, "bold": 1, "medium": 10}
_fonts = {}


def font(size, weight="regular"):
    key = (size, weight)
    if key not in _fonts:
        try:
            _fonts[key] = ImageFont.truetype(_FONT_FILE, size, index=_FACES[weight])
        except OSError:                      # not on a Mac: PIL's built-in font
            _fonts[key] = ImageFont.load_default(size)
    return _fonts[key]


def wrap(draw, text, fnt, max_width, max_lines):
    """Split text into lines that fit max_width; '…' if it needs more than max_lines."""
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=fnt) <= max_width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and draw.textlength(last + "…", font=fnt) > max_width:
            last = last[:-1]
        lines[-1] = last.rstrip() + "…"
    return lines


def fit(draw, text, fnt, max_width):
    lines = wrap(draw, text, fnt, max_width, 1)
    return lines[0] if lines else ""


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def render_camera(frame, objects, hand, target, index_knuckle, index_tip):
    """Just the annotated camera picture (BGR, same size as frame), for the web page.
    Thin boxes, the hand and the pointing ray; the page draws everything else."""
    frame = frame.copy()
    target_label = target["label"] if target else None
    for label, box in objects:
        x1, y1, x2, y2 = map(int, box)
        is_target = label == target_label
        cv2.rectangle(frame, (x1, y1), (x2, y2), YELLOW[::-1] if is_target else (225, 225, 225),
                      3 if is_target else 2, cv2.LINE_AA)
    if hand:
        for x, y in hand:
            cv2.circle(frame, (int(x), int(y)), 3, (210, 210, 210), -1, cv2.LINE_AA)
        (kx, ky), (tx, ty) = index_knuckle, index_tip
        far = (int(tx + (tx - kx) * 20), int(ty + (ty - ky) * 20))
        cv2.line(frame, (int(tx), int(ty)), far, YELLOW[::-1], 2, cv2.LINE_AA)
        cv2.circle(frame, (int(tx), int(ty)), 8, YELLOW[::-1], -1, cv2.LINE_AA)
    if not objects:
        return frame
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    f = font(20, "medium")
    for label, box in objects:
        is_target = label == target_label
        w = draw.textlength(label, font=f) + 22
        x, y = max(6, box[0]), max(6, box[1] - 36)
        draw.rounded_rectangle((x, y, x + w, y + 30), 15, fill=YELLOW if is_target else (30, 30, 34))
        draw.text((x + 11, y + 15), label, font=f, anchor="lm",
                  fill=DARK_TEXT if is_target else TEXT)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


_cam_mask = Image.new("L", (CAM[2] - CAM[0], CAM[3] - CAM[1]), 0)
ImageDraw.Draw(_cam_mask).rounded_rectangle((0, 0, *_cam_mask.size), RADIUS, fill=255)


def render(frame, objects, hand, target, thumbs, index_knuckle, index_tip):
    """Build the whole window picture (BGR numpy, WIDTH x HEIGHT).
    frame: the mirrored camera picture. objects: [(label, box)]. hand: 21 points
    or None. target: current_target() or None. thumbs: None, or (progress 0-1, fired)."""
    s = state.snapshot()
    cam_w, cam_h = CAM[2] - CAM[0], CAM[3] - CAM[1]
    fh, fw = frame.shape[:2]
    scale = max(cam_w / fw, cam_h / fh)                # cover the camera area, crop the rest
    off_x, off_y = (fw * scale - cam_w) / 2, (fh * scale - cam_h) / 2

    def to_canvas(x, y):
        return x * scale - off_x + CAM[0], y * scale - off_y + CAM[1]

    # --- camera side: thin shapes drawn on the frame, text added later with PIL ---
    frame = frame.copy()
    line_w = max(1, round(1.5 / scale))
    target_label = target["label"] if target else None
    for label, box in objects:
        x1, y1, x2, y2 = map(int, box)
        is_target = label == target_label
        color = YELLOW[::-1] if is_target else (225, 225, 225)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_w * (2 if is_target else 1),
                      cv2.LINE_AA)
    if hand:
        for x, y in hand:
            cv2.circle(frame, (int(x), int(y)), max(2, line_w + 1), (210, 210, 210), -1, cv2.LINE_AA)
        (kx, ky), (tx, ty) = index_knuckle, index_tip
        far = (int(tx + (tx - kx) * 20), int(ty + (ty - ky) * 20))
        cv2.line(frame, (int(tx), int(ty)), far, YELLOW[::-1], line_w, cv2.LINE_AA)
        cv2.circle(frame, (int(tx), int(ty)), 7, YELLOW[::-1], -1, cv2.LINE_AA)

    small = cv2.resize(frame, (round(fw * scale), round(fh * scale)), interpolation=cv2.INTER_AREA)
    small = small[round(off_y):round(off_y) + cam_h, round(off_x):round(off_x) + cam_w]
    small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    canvas.paste(Image.fromarray(small), CAM[:2], _cam_mask)
    draw = ImageDraw.Draw(canvas)

    # object labels: small pills above each box
    f_label = font(15, "medium")
    for label, box in objects:
        x, y = to_canvas(box[0], box[1])
        x = max(CAM[0] + 8, x)
        y = max(CAM[1] + 8, y - 26)
        w = draw.textlength(label, font=f_label) + 16
        if x + w > CAM[2] - 8:
            continue
        is_target = label == target_label
        draw.rounded_rectangle((x, y, x + w, y + 22), 11,
                               fill=YELLOW if is_target else (30, 30, 34))
        draw.text((x + 8, y + 11), label, font=f_label, anchor="lm",
                  fill=DARK_TEXT if is_target else TEXT)

    # thumbs-up progress: a pill in the camera's bottom-left corner
    if thumbs:
        progress, fired = thumbs
        text = "thumbs up" if fired else f"thumbs up {progress:.0%}"
        f = font(17, "medium")
        w = draw.textlength(text, font=f) + 28
        x, y = CAM[0] + 16, CAM[3] - 16 - 34
        draw.rounded_rectangle((x, y, x + w, y + 34), 17, fill=GREEN if fired else (30, 30, 34))
        if not fired:
            draw.rounded_rectangle((x, y + 30, x + w * progress, y + 34), 2, fill=GREEN)
        draw.text((x + 14, y + 17), text, font=f, anchor="lm", fill=DARK_TEXT if fired else TEXT)

    # --- right panel ---
    draw.rounded_rectangle(PANEL, RADIUS, fill=PANEL_BG)
    x0, x1 = PANEL[0] + PAD, PANEL[2] - PAD
    inner = x1 - x0
    y = PANEL[1] + PAD

    draw.text((x0, y), "bridge", font=font(34, "bold"), fill=TEXT)
    y += 54

    status = s["status"]
    color = STATUS_COLORS.get(status, MUTED)
    draw.ellipse((x0, y + 5, x0 + 12, y + 17), fill=color)
    draw.text((x0 + 22, y), status, font=font(20, "medium"), fill=color)
    y += 50

    def field(label, value, y):
        draw.text((x0, y), label, font=font(14, "medium"), fill=MUTED)
        y += 22
        f = font(22)
        for line in wrap(draw, value or "–", f, inner, 2):
            draw.text((x0, y), line, font=f, fill=TEXT if value else FAINT)
            y += 28
        return y + 16

    pointing = s["pointing"] if s["pointing_locked"] else target_label
    y = field("heard", s["heard"], y)
    y = field("pointing at", pointing, y)
    y += 6

    # option cards
    bottom_limit = PANEL[3] - PAD - 56
    options = s["options"]
    if not options:
        draw.text((x0, y), "hold space and speak", font=font(18), fill=FAINT)
    f_card, f_num = font(21, "medium"), font(17, "bold")
    for i, option in enumerate(options):
        if i == s["chosen"]:
            bg, fg, num_bg, num_fg = GREEN, DARK_TEXT, (66, 160, 98), DARK_TEXT
        elif i == s["highlight"] and s["chosen"] is None:
            bg, fg, num_bg, num_fg = YELLOW, DARK_TEXT, (222, 190, 86), DARK_TEXT
        else:
            bg, fg, num_bg, num_fg = CARD_BG, TEXT, (62, 62, 70), TEXT
        text_x = x0 + 16 + 30 + 14
        lines = wrap(draw, option, f_card, x1 - 16 - text_x, 3)
        h = max(64, len(lines) * 27 + 30)
        if y + h > bottom_limit:
            break
        draw.rounded_rectangle((x0, y, x1, y + h), 14, fill=bg)
        cy = y + h / 2
        draw.ellipse((x0 + 16, cy - 15, x0 + 46, cy + 15), fill=num_bg)
        draw.text((x0 + 31, cy), str(i + 1), font=f_num, anchor="mm", fill=num_fg)
        ty = cy - len(lines) * 27 / 2
        for line in lines:
            draw.text((text_x, ty + 13), line, font=f_card, anchor="lm", fill=fg)
            ty += 27
        y += h + 12

    # bottom: timings and offline note
    t = s["timings"]
    parts = [f"{name} {t[name] * 1000:.0f} ms" for name in ("stt", "bonsai", "tts") if name in t]
    f_small = font(14)
    draw.text((x0, PANEL[3] - PAD - 40), fit(draw, "  ·  ".join(parts), f_small, inner),
              font=f_small, fill=MUTED)
    draw.text((x0, PANEL[3] - PAD - 18), "offline  ·  wi-fi off", font=f_small, fill=FAINT)

    # --- urgent banner across the whole window ---
    if s["urgent"]:
        draw.rounded_rectangle((MARGIN, MARGIN, WIDTH - MARGIN, MARGIN + 84), RADIUS, fill=RED)
        draw.text((MARGIN + 32, MARGIN + 42), f"urgent: {s['urgent']}", font=font(36, "bold"),
                  anchor="lm", fill=(255, 255, 255))
        note = "caregiver: check now"
        draw.text((WIDTH - MARGIN - 32, MARGIN + 42), note, font=font(20, "medium"),
                  anchor="rm", fill=(255, 220, 220))

    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
