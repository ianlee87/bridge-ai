"""Audio for Bridge: push-to-talk mic recording, WhisperKit speech-to-text, and TTS.

The WhisperKit and argmax-cli calls are copied from ~/lamp-robot/audio.py.
The one new piece is push-to-talk: lamp-robot listens for speech automatically,
but Bridge records only while a key is held down.
"""
import os
import subprocess
import tempfile
import wave

import numpy as np
import sounddevice as sd
from openai import OpenAI

SAMPLE_RATE = 16000          # 16 kHz mono: what Whisper expects

STT_BASE_URL = "http://localhost:50060/v1"   # local WhisperKit server
STT_MODEL = "base.en"

ARGMAX_DIR = os.path.expanduser("~/argmax-oss-swift")
# The prebuilt release binary. Calling it directly skips the slow build check
# that `swift run` does every time. If it's missing, build it once with:
#   cd ~/argmax-oss-swift && swift build -c release
ARGMAX_CLI = os.path.join(ARGMAX_DIR, ".build", "release", "argmax-cli")
TTS_TIMEOUT_S = 30


class PushToTalkRecorder:
    """Call start() when the key goes down and stop() when it comes up.

    While recording, sounddevice calls _on_audio() in the background with
    small chunks of mic audio, and we collect them in a list.
    """

    def __init__(self):
        self.chunks = []
        self.stream = None

    def _on_audio(self, indata, frames, time_info, status):
        self.chunks.append(indata[:, 0].copy())   # keep channel 0 only

    def start(self):
        self.chunks = []
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                     dtype="float32", callback=self._on_audio)
        self.stream.start()

    def stop(self):
        """Stop recording. Returns the audio, or None if it was too short."""
        if self.stream is None:
            return None
        self.stream.stop()
        self.stream.close()
        self.stream = None
        if not self.chunks:
            return None
        audio = np.concatenate(self.chunks)
        if len(audio) < SAMPLE_RATE * 0.3:   # under 0.3 s: probably a key tap
            return None
        return audio


def save_wav(audio):
    """Write float audio (-1..1) to a temporary 16-bit WAV file and return its path."""
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="bridge_")
    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return path


class Transcriber:
    """Talks to the local WhisperKit server, which speaks the OpenAI API."""

    def __init__(self, base_url=STT_BASE_URL, model=STT_MODEL):
        self.client = OpenAI(base_url=base_url, api_key="local")
        self.model = model

    def transcribe_file(self, path):
        """Return (text, words). words is a list of {"word", "start", "end"}
        dicts, or None if the server didn't send word timestamps."""
        with open(path, "rb") as f:
            result = self.client.audio.transcriptions.create(
                model=self.model,
                file=f,
                response_format="verbose_json",       # ask for timing info...
                timestamp_granularities=["word"],     # ...per word
            )
        text = (getattr(result, "text", "") or "").strip()

        words = None
        raw_words = getattr(result, "words", None)
        if raw_words:
            words = []
            for w in raw_words:
                # Each item may be an object or a plain dict depending on the client.
                get = w.get if isinstance(w, dict) else lambda k: getattr(w, k, None)
                if get("word") is not None and get("start") is not None and get("end") is not None:
                    words.append({"word": get("word").strip(),
                                  "start": float(get("start")),
                                  "end": float(get("end"))})
            words = words or None
        return text, words

    def transcribe(self, audio):
        """Same as transcribe_file, but for audio straight from the mic."""
        path = save_wav(audio)
        try:
            return self.transcribe_file(path)
        finally:
            os.remove(path)


def warm_up_tts(timeout_s=60):
    """Run argmax-cli once on the word "ready", silently, and throw the audio away.

    Each speak() starts a new argmax-cli process. We measured the first run
    after a while at ~20 s and later runs at ~5 s (likely macOS caching the
    model files). Doing that slow first run here, at startup, means the
    patient's first real sentence gets the fast path.
    Returns True if it worked."""
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="bridge_warm_")
    os.close(fd)
    try:
        subprocess.run(
            [ARGMAX_CLI, "tts", "--text", "ready",
             "--output-path", path[:-4], "--output-format", "wav"],   # no --play: silent
            cwd=ARGMAX_DIR, check=True, timeout=timeout_s,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.SubprocessError, OSError):
        return False
    finally:
        if os.path.exists(path):
            os.remove(path)


def speak(text, timeout_s=TTS_TIMEOUT_S):
    """Speak text out loud (waits until done). Returns which engine was used.
    Same arguments as lamp-robot: argmax-cli first, macOS `say` if that fails."""
    try:
        subprocess.run(
            [ARGMAX_CLI, "tts", "--text", text, "--play"],
            cwd=ARGMAX_DIR, check=True, timeout=timeout_s,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return "argmax"
    except (subprocess.SubprocessError, OSError) as e:
        print(f"[tts] argmax failed ({type(e).__name__}), using macOS say")
        subprocess.run(["say", text], check=False)
        return "say"
