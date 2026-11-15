# Bridge: offline speech companion for aphasia (Bonsai + Argmax)

Bridge listens to a person with aphasia, turns their broken phrases into up to
3 sentences they probably meant, and speaks the one they pick.
Everything runs on this Mac. It works with Wi-Fi off.

```
hold SPACE -> WhisperKit (speech to text) -> split on pauses -> urgent check
           -> + what the camera sees the patient pointing at
           -> Bonsai (3 options) -> first-person filter
           -> patient picks 1/2/3 (or thumbs-up) -> argmax-cli speaks it
```

## Files

| File | What it does |
|---|---|
| `main.py` | The program: keyboard, screen display, and the main loop |
| `audio.py` | Push-to-talk recording, WhisperKit speech-to-text, argmax-cli speech |
| `fragments.py` | Splits words into fragments at pauses > 0.6 s; urgent keyword rule |
| `brain.py` | Bonsai prompt, 5 worked examples, JSON parsing, first-person filter, warm-up |
| `vision.py` | Webcam + YOLO + MediaPipe hands: pointing target, thumbs-up, and the camera window drawing. Run alone to test just vision |
| `models/hand_landmarker.task` | MediaPipe hand model (not in git; see setup) |
| `eval.py` | Scores Bonsai on the 15 test cases from text (no audio needed) |
| `test_phrases.txt` | 15 scored test cases: caregiver question, fragments, pointed-at object, expected sentence |

## Setup (one time)

**1. Python environment**

```bash
cd ~/bridge
python3 -m venv .venv
.venv/bin/pip install openai sounddevice numpy rich pynput ultralytics mediapipe opencv-python
```

Vision needs two model files in `~/bridge` (both are kept out of git):
- `yolov8n.pt`: the YOLO object model
- `models/hand_landmarker.task`: the MediaPipe hand model. Copy it from lamp-vision:
  `mkdir -p models && cp ~/lamp-vision/models/hand_landmarker.task models/`

Bridge sets `YOLO_OFFLINE=1`, so YOLO never tries the internet.

**2. Argmax speech binary.** Bridge calls the prebuilt
`~/argmax-oss-swift/.build/release/argmax-cli` directly. That's faster than
`swift run`, which checks the build every time. If that file doesn't exist, build it once:

```bash
cd ~/argmax-oss-swift && swift build -c release
```

**3. LM Studio.** Download and load the model `ternary-bonsai-2-4b-mlx`, then start
the local server on port **1235** (Developer tab > server port 1235 > Start).

**4. macOS permissions.** Open System Settings > Privacy & Security and turn on
your terminal app (Terminal, iTerm, VS Code…) under:
- **Input Monitoring**: lets Bridge see SPACE and C being held down and released
- **Accessibility**: also needed by the keyboard library on macOS
- **Microphone**: to record
- **Camera**: for pointing and thumbs-up (skip with `--no-vision`)

Quit and reopen the terminal after changing these. If Input Monitoring is off,
Bridge prints a yellow warning and the keys do nothing.

## Every time: start the servers

1. **WhisperKit** (speech to text, port 50060), in its own terminal window:
   ```bash
   cd ~/argmax-oss-swift && BUILD_ALL=1 swift run argmax-cli serve --model base.en --verbose
   ```
   It's ready when `curl localhost:50060/v1/models` stops saying
   "connection refused" (about 25 s).
2. **LM Studio** server on port 1235 with Bonsai loaded (see setup step 3).

## Live mode

```bash
cd ~/bridge
.venv/bin/python main.py               # camera 0, with the camera window
.venv/bin/python main.py --camera 1    # use a different camera
.venv/bin/python main.py --no-window   # camera on, but no window
.venv/bin/python main.py --no-vision   # no camera at all
```

By default a **camera window** opens next to the terminal. It shows the object
boxes, the fingertip and its ray, and `pointing at: X (method)` along the top.
While the patient is picking, the options are drawn along the bottom with the
current one highlighted in yellow. Keys still work while the window is in front.

How it works: macOS only lets the main thread of a program draw windows. So
with the window on, the main thread runs the camera and the window, and the
keyboard + voice loop runs in a background thread. With `--no-window` it's the
other way round: the voice loop is on the main thread and the camera runs in
the background.

If the camera can't open, Bridge prints a yellow warning and keeps running
without vision or window. If the camera stops partway through, the window
closes and Bridge keeps running without vision. On start, Bridge sends Bonsai a tiny request so the model is loaded (**Bonsai warm**),
then silently runs argmax-cli once on the word "ready" so the first spoken
sentence isn't slow (**Speech warm**). Both print how long they took.

| Key | Action |
|---|---|
| hold **SPACE** | Record the patient. Release to stop. |
| hold **C** | Record the caregiver's question. It's given to Bonsai as context for the patient's next turn. |
| **1 / 2 / 3** | Speak that option out loud |
| **0** | None of these, try again (nothing is spoken) |
| **Q** / **Esc** | Quit |

Nothing is ever spoken unless the patient picks: key 1, 2 or 3, or a thumbs-up.

### Pointing (vision)

While the patient speaks, they can point at something (cup, remote, tv, book…).
When SPACE is released, Bridge takes what the camera thinks they're pointing at
and gives it to Bonsai ("Patient is pointing at: cup"). The terminal shows
`Pointing at: cup (ray)` and a `vision` time. The method says how it was found:

| Method | Meaning |
|---|---|
| `ray` | A line from the index knuckle through the fingertip hits that object first |
| `nearest` | The line hit nothing, so it's the object closest to the fingertip (within 150 px) |
| `held` | No hand visible, so it's the biggest object near the middle of the picture (held up to the camera) |

An object only counts after it has been the target for about 0.5 s, and it is
forgotten 2 s after it stops being seen. People are never a target.

### Thumbs-up to pick

With vision on, after the options appear a yellow highlight starts on option 1
and moves to the next option every 2.5 s. Hold a thumbs-up for about 0.7 s
to pick the highlighted option. The keys 1/2/3/0 still work at any time.
Only a thumbs-up that starts after the options appear counts.

### Testing vision alone

```bash
.venv/bin/python vision.py             # debug window, q = quit
.venv/bin/python vision.py --camera 1
```

You should see green boxes with labels around objects. A pink dot marks your
index fingertip, and a pink line runs from the knuckle through the fingertip.
The target's box turns yellow, and the top bar says e.g. `pointing at: cup (ray)`.
Hold a thumbs-up and `thumbs up 50%` counts up to `THUMBS UP!`. The terminal
prints the FPS every 2 s and each change of target. If the picture is black,
try `--camera 1`.

### First-person filter

After Bonsai answers, options that sound like someone talking *to* the patient
are dropped:
1. they ask the patient something: start with "Do you", "Would you like" or "Are you"
2. they say "your" (the patient's), e.g. "Is it your son?"
3. **kept:** requests to the caregiver that start with "Could you", "Would you",
   "Can you" or "Please", e.g. "Could you change the channel?"
4. otherwise they're dropped if they contain "you" but no "I/me/my"

The checks run in that order. So "Could I have a blanket?", "Can you turn that down?"
and "Would you give me that hot drink?" are kept, while "Do you want cold water?"
and "Would you like a drink?" are dropped.

The filter is skipped if the patient's fragments contain
who/what/where/when/why/how. Whisper's "?" is ignored, because it adds one
whenever the voice rises ("Want?"). If fewer than 2 options survive, Bonsai is
asked once more. So sometimes only 1 or 2 options are shown.

### Urgent alert

If the fragments contain **"chest" + "pain"**, **"can't breathe"** (also
"cant"/"cannot"), or **"help" on its own** ("help", "help me", "help, please"),
Bridge skips the AI and shows a red URGENT box for the caregiver.
"bathroom… help" is a normal request (it has another content word), and
"helpful" never counts. This is a plain keyword check, not the AI, so it is
instant and predictable. It is only a prompt to look at the patient. It is
**not** a medical alarm and does not call anyone.

## --wav test mode (no keyboard or mic)

Runs one turn from a WAV file. You can make one with macOS `say`;
`[[slnc 1000]]` inserts a 1-second pause:

```bash
say -o test.wav --data-format=LEI16@16000 "water [[slnc 1000]] want [[slnc 1000]] cold"
.venv/bin/python main.py --wav test.wav --pick 1
.venv/bin/python main.py --wav test.wav --caregiver "Are you thirsty?" --pointing cup
```

With no `--pick`, nothing is spoken. `--pointing` stands in for the future camera.

## eval.py: score Bonsai on the test cases

```bash
.venv/bin/python eval.py           # ask Bonsai about every case, then judge them
.venv/bin/python eval.py --judge   # judge saved results again, no Bonsai calls
```

Only LM Studio is needed (no WhisperKit, no audio). Run it in a terminal.

1. **Generate:** each case in `test_phrases.txt` goes through the urgent check,
   then Bonsai plus the filter. The 14 normal cases run twice, **with** and
   **without** `pointing_at`. Everything is saved to `eval_results.csv`.
2. **Judge:** for each row you see the expected sentence and the options, and type
   **y** (an option says what the patient meant), **n**, **s** (skip) or
   **q** (save and quit). Each answer is saved right away. `--judge` only asks
   about rows that aren't marked yet; to redo one, clear its `judge` cell in the CSV.
   When you rerun eval.py, rows whose options didn't change keep your old y/n
   (same options in a different order count as unchanged). Changed rows are
   printed with `-` old / `+` new lines and need judging again.
   The URGENT case is judged automatically (did the alert fire?).
3. **Summary:** your hit rate with vs. without pointing, side by side, plus the
   keyword-overlap score as a rough second column, and average Bonsai time.

The keyword score counts shared words, not meaning. "I'm not taking this apple"
scores well for "I'd like the apple", so trust your y/n over it.
