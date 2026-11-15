"""Bridge's local web UI: http://localhost:8000 (run `python main.py --web`).

Fully offline: the page, its CSS and JS are files in static/, nothing from the internet.
The web side only shows state and sends picks; all the real work stays in main.py.

    /          the page (static/index.html)
    /video     the annotated camera picture as MJPEG (made by vision's own thread)
    /events    live state as server-sent events (JSON, sent whenever it changes)
    /state     the same state once, as JSON (handy for checks)
    /pick      POST {"n": 1}: a click on option card n. Becomes the same key press
               as pressing 1 / 2 / 3, so main.py's picking logic is unchanged.
"""
import json
import logging
import os
import threading
import time

from flask import Flask, Response, jsonify, request, send_from_directory

from ui import state

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = Flask(__name__, static_folder=STATIC, static_url_path="/static")
_vision = None
_keys = None


def current_state():
    s = state.snapshot()
    locked = s.pop("pointing_locked")
    live = {"camera": _vision is not None, "live_pointing": None, "live_method": None,
            "thumbs": None}
    if _vision is not None:
        target = _vision.current_target()
        if target:
            live["live_pointing"], live["live_method"] = target["label"], target["method"]
        thumbs = _vision.get_thumbs_up_progress()
        if thumbs:
            live["thumbs"] = {"progress": round(thumbs[0], 2), "fired": thumbs[1]}
    # During a turn show what was captured for it; otherwise what the camera sees now.
    s["pointing_at"] = s["pointing"] if locked else live["live_pointing"]
    s["pointing_method"] = s["pointing_method"] if locked else live["live_method"]
    s["timings_ms"] = {k: round(v * 1000) for k, v in s.pop("timings").items()}
    s.update(live)
    return s


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/state")
def get_state():
    return jsonify(current_state())


@app.get("/events")
def events():
    def stream():
        last, last_sent = None, 0.0
        while True:
            data = json.dumps(current_state())
            if data != last:
                last, last_sent = data, time.time()
                yield f"data: {data}\n\n"
            elif time.time() - last_sent > 15:        # keep-alive comment
                last_sent = time.time()
                yield ": ping\n\n"
            time.sleep(0.08)
    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/video")
def video():
    if _vision is None:
        return Response("camera off", status=404)

    def stream():
        seen = -1
        while True:
            with _vision.lock:
                jpeg, number = _vision.jpeg, _vision.jpeg_no
            if jpeg is not None and number != seen:
                seen = number
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(0.02)
    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache"})


@app.post("/pick")
def pick():
    s = state.snapshot()
    try:
        n = int((request.get_json(silent=True) or {}).get("n"))
    except (TypeError, ValueError):
        return jsonify(ok=False, error="send {\"n\": 0-3}"), 400
    if s["status"] != "picking" or not 0 <= n <= len(s["options"]):
        return jsonify(ok=False, error="nothing to pick right now"), 409
    _keys.events.put(("down", str(n)))       # exactly what pressing that key does
    _keys.events.put(("up", str(n)))
    return jsonify(ok=True)


def start(vision, keys, port=8000):
    """Run the server in a background thread on localhost only."""
    global _vision, _keys
    _vision, _keys = vision, keys
    logging.getLogger("werkzeug").setLevel(logging.ERROR)    # no line per request
    import flask.cli
    flask.cli.show_server_banner = lambda *a, **k: None
    threading.Thread(target=lambda: app.run(host="127.0.0.1", port=port, threaded=True,
                                            use_reloader=False),
                     daemon=True).start()
