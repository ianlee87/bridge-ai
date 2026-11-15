"""Bridge: offline speech companion for people with aphasia.

Live mode (normal use):
    python main.py                 (shows the camera window; --camera 1 for another camera,
                                    --no-window to hide the window, --no-vision for no camera)
      hold SPACE  = record the patient, release to stop
      (the camera notes what the patient points at; thumbs-up picks the highlighted option)
      hold C      = record the caregiver's question (used as context next turn)
      1 / 2 / 3   = speak that option,  0 = none of these, try again
      Q or Esc    = quit

Test mode (no keyboard or mic needed; used for end-to-end checks):
    python main.py --wav clip.wav --pick 1 [--caregiver "Are you hungry?"] [--pointing cup]
"""
import argparse
import queue
import sys
import termios
import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from audio import SAMPLE_RATE, PushToTalkRecorder, Transcriber, speak, warm_up_tts
from brain import LLM_MODEL, Brain
from fragments import check_urgent, make_fragments

console = Console()


def ms(seconds):
    return f"{seconds * 1000:.0f} ms"


# ---------------------------------------------------------------------------
# One patient turn: transcribe -> fragments -> urgent check -> Bonsai -> show
# ---------------------------------------------------------------------------

def warm_up(brain):
    """Get Bonsai and the speech engine ready now, so the patient's first turn isn't slow."""
    t0 = time.perf_counter()
    try:
        brain.warm_up()
        console.print(f"[green]Bonsai warm[/] [dim]({ms(time.perf_counter() - t0)})[/]")
    except Exception as e:
        console.print(f"[red]Bonsai warm-up failed: {e}. Is LM Studio running on port 1235?[/]")

    t0 = time.perf_counter()
    if warm_up_tts():
        console.print(f"[green]Speech warm[/] [dim]({ms(time.perf_counter() - t0)})[/]")
    else:
        console.print("[yellow]Speech warm-up failed; the first sentence may be slow.[/]")


def run_patient_turn(stt, brain, audio=None, wav_path=None, caregiver_question=None,
                     pointing_at=None, pointing_method=None, vision_seconds=None):
    """Process one recording. Returns (options, timings) or (None, timings) if
    there is nothing to choose from (empty transcript or URGENT alert).
    pointing_at/pointing_method come from vision (or --pointing in test mode)."""
    timings = {}
    if vision_seconds is not None:
        timings["vision"] = vision_seconds
        where = f"{pointing_at} ({pointing_method})" if pointing_at else "nothing"
        console.print(f"[bold]Pointing at:[/] {where}")

    # 1) Speech to text
    t0 = time.perf_counter()
    if wav_path:
        text, words = stt.transcribe_file(wav_path)
    else:
        text, words = stt.transcribe(audio)
    timings["speech-to-text (WhisperKit)"] = time.perf_counter() - t0

    # 2) Fragments: split on pauses if we got word timestamps, else raw text
    t0 = time.perf_counter()
    fragments, method = make_fragments(text, words)
    timings["fragments"] = time.perf_counter() - t0

    console.print(f"[bold]Raw transcript:[/] {text or '[dim](nothing heard)[/]'}")
    console.print(f"[bold]Fragment method:[/] {method}")
    if words:
        console.print("[dim]Word timings: " + "  ".join(
            f"{w['word']}({w['start']:.2f}-{w['end']:.2f})" for w in words) + "[/]")
    console.print("[bold]Fragments:[/] " + "  |  ".join(f"[cyan]{f}[/]" for f in fragments))

    if not fragments:
        console.print("[yellow]Nothing heard. Try again.[/]")
        show_timings(timings)
        return None, timings

    # 3) Urgent keyword rule: plain Python, runs BEFORE the AI and skips it
    t0 = time.perf_counter()
    reason = check_urgent(fragments)
    timings["urgent check"] = time.perf_counter() - t0
    if reason:
        console.print(Panel(
            f"[bold white]The patient may need help right now.[/]\n\n"
            f"Heard: \"{' '.join(fragments)}\"\nTrigger: {reason}",
            title="[bold]URGENT: CAREGIVER CHECK NOW[/]",
            style="bold white on red", padding=(1, 2)))
        show_timings(timings)
        return None, timings

    # 4) Bonsai suggests 3 sentences (then the first-person filter runs)
    if caregiver_question:
        console.print(f"[bold]Context:[/] Caregiver just asked: \"{caregiver_question}\"")
    if pointing_at and vision_seconds is None:      # (vision already printed it above)
        console.print(f"[bold]Context:[/] Patient is pointing at: {pointing_at}")
    t0 = time.perf_counter()
    try:
        result = brain.suggest(fragments, caregiver_question, pointing_at)
    except Exception as e:
        console.print(f"[red]Bonsai failed: {e}. Is LM Studio running on port 1235?[/]")
        return None, timings
    timings["Bonsai options" + (" (asked twice)" if result["retried"] else "")] = \
        time.perf_counter() - t0
    options = result["options"]

    for d in result["dropped"]:
        console.print(f"[dim]Filtered out (not first person): {d}[/]")
    if not options:
        console.print("[yellow]No good options this time. Please try again.[/]")
        show_timings(timings)
        return None, timings

    console.print(f"[bold]Options[/] [dim](parsed via {result['how']})[/]:")
    for i, option in enumerate(options, start=1):
        console.print(f"  [bold green]{i}[/]  {option}")
    console.print("  [bold yellow]0[/]  None of these, try again")
    show_timings(timings)
    return options, timings


def show_timings(timings):
    table = Table(title="Time per stage", show_header=True, header_style="bold")
    table.add_column("Stage")
    table.add_column("Time", justify="right")
    for stage, seconds in timings.items():
        table.add_row(stage, ms(seconds))
    console.print(table)


def speak_choice(options, pick):
    """Speak option number `pick` (1-3). Returns the TTS time in seconds."""
    sentence = options[pick - 1]
    console.print(f"[bold green]Speaking:[/] {sentence}")
    t0 = time.perf_counter()
    engine = speak(sentence)
    seconds = time.perf_counter() - t0
    console.print(f"[dim]text-to-speech ({engine}): {ms(seconds)}[/]")
    return seconds


# ---------------------------------------------------------------------------
# Test mode: one turn from a WAV file, pick given on the command line
# ---------------------------------------------------------------------------

def run_test(args):
    stt, brain = Transcriber(), Brain()
    console.rule("[bold]Bridge test turn")
    warm_up(brain)
    console.print(f"[dim]Audio file: {args.wav}[/]")
    options, _ = run_patient_turn(stt, brain, wav_path=args.wav,
                                  caregiver_question=args.caregiver,
                                  pointing_at=args.pointing)
    if not options:
        return
    if args.pick == 0:
        console.print("[yellow]Pick 0: none of these. Nothing spoken.[/]")
    elif args.pick and 1 <= args.pick <= len(options):
        console.print(f"[bold]Pick:[/] {args.pick}")
        speak_choice(options, args.pick)
    else:
        console.print("[yellow]No --pick given, so nothing is spoken.[/]")


# ---------------------------------------------------------------------------
# Live mode: keyboard + microphone
# ---------------------------------------------------------------------------

def key_name(key):
    """Turn a pynput key into a simple string like "space", "c", "1"."""
    from pynput import keyboard
    if key == keyboard.Key.space:
        return "space"
    if key == keyboard.Key.esc:
        return "esc"
    char = getattr(key, "char", None)
    return char.lower() if char else None


class Keys:
    """Listens to the keyboard in the background and queues ("down"/"up", name)
    events. Holding a key makes the OS repeat "down" events; we drop those."""

    def __init__(self):
        from pynput import keyboard
        self.events = queue.Queue()
        self.held = set()
        self.listener = keyboard.Listener(on_press=self._down, on_release=self._up)
        self.listener.start()
        self.trusted = getattr(self.listener, "IS_TRUSTED", True)

    def _down(self, key):
        name = key_name(key)
        if name and name not in self.held:
            self.held.add(name)
            self.events.put(("down", name))

    def _up(self, key):
        name = key_name(key)
        if name:
            self.held.discard(name)
            self.events.put(("up", name))

    def clear(self):
        """Throw away keys pressed while we were busy (thinking or speaking)."""
        while not self.events.empty():
            self.events.get_nowait()

    def wait_down(self, allowed, timeout=None):
        """Wait until one of the `allowed` keys is pressed and return its name.
        With a timeout (seconds), return None if no allowed key came in time."""
        deadline = None if timeout is None else time.time() + timeout
        while True:
            remaining = None if deadline is None else deadline - time.time()
            if remaining is not None and remaining <= 0:
                return None
            try:
                kind, name = self.events.get(timeout=remaining)
            except queue.Empty:
                return None
            if kind == "down" and name in allowed:
                return name

    def wait_up(self, name):
        while True:
            kind, n = self.events.get()
            if kind == "up" and n == name:
                return


def record_while_held(keys, recorder, key, label):
    """Record from the mic until `key` is released."""
    recorder.start()
    t0 = time.perf_counter()
    console.print(f"[bold red]● Recording {label}...[/] (release {key.upper()} to stop)")
    keys.wait_up(key)
    audio = recorder.stop()
    console.print(f"[dim]recorded {ms(time.perf_counter() - t0)}[/]")
    return audio


HIGHLIGHT_STEP_S = 2.5    # thumbs-up mode: highlight moves to the next option this often


WINDOW_TITLE = "Bridge camera"


def start_vision(camera_index, background=True):
    """Open the camera and load the vision models. Returns the Vision object, or
    None if it failed. background=True also starts vision's own thread; with
    background=False the caller must call vision.step() itself (window mode)."""
    console.print("[dim]Starting camera...[/]")
    try:
        from vision import Vision      # imported here so --no-vision never loads YOLO
        vision = Vision(camera_index)
        ok, message = vision.start() if background else vision.open()
    except Exception as e:
        ok, message = False, f"{type(e).__name__}: {e}"
    if not ok:
        console.print(f"[bold yellow]Vision is off: {message}[/]\n"
                      "Bridge will keep running without pointing or thumbs-up.")
        return None
    console.print(f"[green]Vision on[/] [dim]({message})[/]")
    return vision


def wait_for_pick(keys, options, vision):
    """Wait for the patient's pick. Returns (number, how): 0 = try again.
    Keys 1/2/3/0 always work. With vision on, a highlight steps through the
    options every HIGHLIGHT_STEP_S seconds and a thumbs-up picks the highlighted one."""
    keys.clear()
    allowed = {"0"} | {str(i) for i in range(1, len(options) + 1)}
    if vision is None:
        console.print("[bold]Pick 1, 2, 3, or 0 to try again.[/]")
        return int(keys.wait_down(allowed)), "key"

    console.print("[bold]Pick 1, 2, 3, or 0 to try again. Or thumbs-up on the highlighted one.[/]")
    thumbs_before = vision.get_thumbs_up_count()   # only a NEW thumbs-up counts
    highlight = 0                                  # index into options (0 = option 1)
    next_step = time.time() + HIGHLIGHT_STEP_S

    def line():
        return f"[black on yellow] ▶ {highlight + 1}  {options[highlight]} [/]  [dim]thumbs-up to pick[/]"

    # The highlight is shown in two places: this terminal line, and the camera
    # window (if it's open), where the options are drawn at the bottom.
    vision.set_pick_overlay(options, highlight)
    try:
        with Live(line(), console=console, refresh_per_second=10, transient=True) as live:
            while True:
                key = keys.wait_down(allowed, timeout=0.1)
                if key is not None:
                    return int(key), "key"
                if vision.get_thumbs_up_count() > thumbs_before:
                    return highlight + 1, "thumbs-up"
                if time.time() >= next_step:
                    highlight = (highlight + 1) % len(options)
                    next_step = time.time() + HIGHLIGHT_STEP_S
                    live.update(line())
                    vision.set_pick_overlay(options, highlight)
    finally:
        vision.set_pick_overlay(None, 0)     # picking is over: hide the options


def run_live(vision, keys):
    """The keyboard + voice loop. `vision` may be None (no camera)."""
    stt, brain, recorder = Transcriber(), Brain(), PushToTalkRecorder()
    warm_up(brain)
    if not keys.trusted:
        console.print("[bold yellow]macOS has not allowed this terminal to read the keyboard.[/]\n"
                      "Open System Settings > Privacy & Security > Input Monitoring "
                      "(and Accessibility), turn on your terminal app, then restart Bridge.")

    caregiver_question = None
    console.print(Panel(
        f"[bold]Bridge[/] is ready  (model: {LLM_MODEL})\n"
        "Hold [bold]SPACE[/] = patient speaks     Hold [bold]C[/] = caregiver asks\n"
        "[bold]Q[/] or [bold]Esc[/] = quit", style="blue"))

    while True:
        console.print("\n[bold]Waiting...[/] hold SPACE (patient) or C (caregiver)")
        keys.clear()
        key = keys.wait_down({"space", "c", "q", "esc"})
        if key in ("q", "esc"):
            return

        # ----- Caregiver mode: record the question and remember it -----
        if key == "c":
            audio = record_while_held(keys, recorder, "c", "CAREGIVER")
            if audio is None:
                console.print("[yellow]Too short, try again.[/]")
                continue
            t0 = time.perf_counter()
            text, _ = stt.transcribe(audio)
            console.print(f"[bold cyan]Caregiver asked:[/] {text}  [dim]({ms(time.perf_counter() - t0)})[/]")
            caregiver_question = text or None
            continue

        # ----- Patient turn -----
        audio = record_while_held(keys, recorder, "space", "PATIENT")
        if audio is None:
            console.print("[yellow]Too short, try again.[/]")
            continue
        console.print(f"[dim]audio length: {len(audio) / SAMPLE_RATE:.1f} s[/]")

        # What was the patient pointing at when they let go of SPACE?
        pointing_at = pointing_method = vision_seconds = None
        if vision:
            t0 = time.perf_counter()
            target = vision.current_target()
            vision_seconds = time.perf_counter() - t0
            if target:
                pointing_at, pointing_method = target["label"], target["method"]
        try:
            options, _ = run_patient_turn(stt, brain, audio=audio,
                                          caregiver_question=caregiver_question,
                                          pointing_at=pointing_at,
                                          pointing_method=pointing_method,
                                          vision_seconds=vision_seconds)
        except Exception as e:
            console.print(f"[red]Speech-to-text failed: {e}. Is WhisperKit running on port 50060?[/]")
            continue

        if not options:
            # Empty transcript or URGENT alert: wait for the caregiver to acknowledge.
            console.print("[dim]Press any number key to continue.[/]")
            keys.clear()
            keys.wait_down({"0", "1", "2", "3", "space"})
            continue

        # Patient picks (key or thumbs-up). Nothing is spoken without a pick.
        pick, how = wait_for_pick(keys, options, vision)
        if pick == 0:
            console.print("[yellow]OK, let's try again.[/]")
            continue        # keep the caregiver question for the retry
        console.print(f"[bold]Picked {pick}[/] [dim](by {how})[/]")
        speak_choice(options, pick)
        caregiver_question = None   # that question has been answered


def run_with_window(camera_index, keys):
    """Live mode with the camera window.

    macOS only lets the MAIN thread draw windows, so here the jobs are swapped:
      main thread       - camera + YOLO + hands each frame, and cv2.imshow()
      background thread - the keyboard + voice loop (run_live)
    The two share the Vision object (its values are protected by a lock)."""
    import threading

    import cv2
    from vision import MAX_FPS

    vision = start_vision(camera_index, background=False)
    if vision is None:                    # no camera: same as --no-vision
        run_live(None, keys)
        return

    voice_done = threading.Event()

    def voice_thread():
        try:
            run_live(vision, keys)
        except Exception as e:            # show the error instead of dying silently
            console.print(f"[red]Bridge stopped: {type(e).__name__}: {e}[/]")
        finally:
            voice_done.set()              # tells the main thread to close the window

    threading.Thread(target=voice_thread, daemon=True).start()

    try:
        while not voice_done.is_set():
            started = time.time()
            picture = vision.step(draw=True)
            if picture is None:
                console.print("[bold yellow]The camera stopped sending pictures. "
                              "Closing the window; Bridge keeps running without vision.[/]")
                break
            cv2.imshow(WINDOW_TITLE, picture)
            cv2.waitKey(1)                # lets macOS actually draw the window
            time.sleep(max(0.0, 1 / MAX_FPS - (time.time() - started)))
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        vision.cap.release()
    # If the camera died, keep waiting for the voice loop (Q / Esc / Ctrl+C ends it).
    while not voice_done.wait(0.2):
        pass


def main():
    parser = argparse.ArgumentParser(description="Bridge: aphasia speech companion")
    parser.add_argument("--wav", help="test mode: process this WAV file instead of the mic")
    parser.add_argument("--pick", type=int, help="test mode: option to speak (1-3, or 0)")
    parser.add_argument("--caregiver", help="test mode: caregiver question for context")
    parser.add_argument("--pointing", help="test mode: object the patient points at, e.g. cup")
    parser.add_argument("--no-vision", action="store_true", help="live mode: don't use the camera")
    parser.add_argument("--camera", type=int, default=0, help="live mode: camera number (default 0)")
    parser.add_argument("--no-window", action="store_true",
                        help="live mode: use the camera but don't show its window")
    args = parser.parse_args()

    if args.wav:
        run_test(args)
        return

    # Turn off terminal echo so held keys don't spray spaces/letters on screen.
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] &= ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSADRAIN, new)
    try:
        # The keyboard listener is created here, on the main thread, in every mode.
        keys = Keys()
        if args.no_vision:
            run_live(None, keys)
        elif args.no_window:
            run_live(start_vision(args.camera, background=True), keys)
        else:
            run_with_window(args.camera, keys)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcflush(fd, termios.TCIFLUSH)            # discard typed keys
        termios.tcsetattr(fd, termios.TCSADRAIN, old)    # restore echo
        console.print("\nBridge closed.")


if __name__ == "__main__":
    main()
