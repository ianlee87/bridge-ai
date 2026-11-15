# bridge

an offline speech bridge for people with aphasia.
you say a few words and point at something. it suggests full sentences.
you pick one, and it speaks it.

## how it works

- whisperkit turns your speech into text
- the text gets split into fragments at pauses
- the camera sees what you're pointing at (mediapipe for the hand, yolo for the object)
- bonsai 2 4b (`ternary-bonsai-2-4b-mlx` in lm studio) suggests up to 3 sentences
- you pick one
- argmax tts says it out loud

nothing is spoken until you pick.

## offline

it all runs on the mac. it works with wi-fi off.

## setup

1. python 3.12 venv:
   ```bash
   cd ~/bridge
   python3.12 -m venv .venv
   .venv/bin/pip install openai sounddevice numpy rich pynput ultralytics mediapipe opencv-python
   ```
2. lm studio: load `ternary-bonsai-2-4b-mlx` and start the server on port 1235.
3. whisperkit server on port 50060, in its own terminal tab:
   ```bash
   cd ~/argmax-oss-swift && BUILD_ALL=1 swift run argmax-cli serve --port 50060 --model base.en --verbose
   ```
   it's ready when `curl localhost:50060/v1/models` answers.
4. speech binary, built once:
   ```bash
   cd ~/argmax-oss-swift && swift build -c release
   ```
5. vision models, both go in `~/bridge` and stay out of git:
   - `yolov8n.pt`: from https://github.com/ultralytics/assets/releases (the yolov8n.pt asset)
   - `models/hand_landmarker.task`: from https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
6. macos settings, privacy & security: give your terminal input monitoring, accessibility, microphone and camera. then reopen the terminal.

## run it

```bash
.venv/bin/python main.py --camera 1
```

- hold `space`: record yourself
- hold `c`: record the caregiver's question
- `1` / `2` / `3`: speak that option
- `0`: none of these
- thumbs up: picks the highlighted option (it moves every 2.5 s)
- `q` or `esc`: quit

`--no-vision` turns the camera off. `--no-window` hides the camera window.

"chest pain", "can't breathe" or just "help" skips the model and shows an urgent box.

## eval

```bash
.venv/bin/python eval.py
```

14 self-written cases plus 1 urgent case, judged by hand.
13/14 with pointing. 11/14 without. the urgent case fires.

## limits

- the test speech was acted, not from real patients
- english only
- not a medical device
- the test set is small and i wrote it myself
