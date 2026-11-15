"""Turn a transcript into fragments, and check fragments for urgent words.

Everything here is plain Python with no AI, so it is fast and predictable.
"""
import re

PAUSE_S = 0.6   # a silence longer than this between two words starts a new fragment


def split_on_pauses(words, pause_s=PAUSE_S):
    """words: list of {"word", "start", "end"} from WhisperKit.
    Returns a list of fragment strings, e.g. ["water.", "want?", "cold water."]."""
    fragments = []
    current = []
    previous_end = None
    for w in words:
        # Long gap since the last word ended? Close off the current fragment.
        if previous_end is not None and w["start"] - previous_end > pause_s and current:
            fragments.append(" ".join(current))
            current = []
        current.append(w["word"])
        previous_end = w["end"]
    if current:
        fragments.append(" ".join(current))
    return fragments


def make_fragments(text, words):
    """Returns (fragments, method). method says which path we took."""
    if words:
        return split_on_pauses(words), "word timestamps (pauses > 0.6s)"
    return [text] if text else [], "raw text (no word timestamps)"


# ---------- Urgent keyword rule (no LLM) ----------

def _normalize(text):
    """Lowercase, straighten curly apostrophes, and drop punctuation except apostrophes."""
    text = text.lower().replace("’", "'")
    return re.sub(r"[^a-z' ]+", " ", text)


# Words allowed next to "help" without making it an ordinary request.
# "help" / "help me" / "help, help, please" = URGENT.
# "bathroom... help" has a content word ("bathroom"), so it is a normal request.
HELP_ONLY_WORDS = {"help", "me", "please", "oh", "uh", "um", "ah"}


def check_urgent(fragments):
    """Return a short reason string if the patient may need urgent help, else None."""
    text = " " + _normalize(" ".join(fragments)) + " "
    words = set(text.split())

    if "chest" in words and "pain" in words:
        return "chest + pain"
    # Whisper may write "can't", "cant", "cannot" or "can not", so check them all.
    for phrase in ("can't breathe", "cant breathe", "cannot breathe", "can not breathe"):
        if phrase in text:
            return "can't breathe"
    # "help" only counts when it's said alone (or as "help me"), with nothing else.
    # Whole-word matching means "helpful" never counts.
    if "help" in words and words <= HELP_ONLY_WORDS:
        return "help"
    return None
