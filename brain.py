"""Bonsai (in LM Studio) turns fragments into 3 sentence options.

The client setup and the "content or reasoning_content" trick are copied from
~/lamp-robot/llm.py.
"""
import json
import re

from openai import OpenAI

LLM_BASE_URL = "http://localhost:1235/v1"     # LM Studio
LLM_MODEL = "ternary-bonsai-2-4b-mlx"

SYSTEM_PROMPT = (
    "You help a person with aphasia. Turn their fragments into 3 short first-person "
    "sentences they likely meant, each under 15 words. Never add facts they didn't give. "
    'Reply ONLY as JSON: {"options": ["...", "...", "..."]}'
)

# Worked examples: (caregiver question, fragments, pointing at, expected sentence).
# None means "not given". The expected sentence is always option 1. Options 2
# and 3 are close rewordings, because the reply format needs 3 options.
EXAMPLES = [
    ("What do you need?", ["cold", "please"], "bottle",
     ["Could I have that bottle of cold water, please?",
      "I'd like that cold bottle, please.",
      "Please give me the cold bottle."]),
    ("Are you hungry?", ["no", "tired"], None,
     ["No, I'm not hungry. I'm just tired.",
      "No, I'm tired, not hungry.",
      "Not hungry. I'm tired."]),
    (None, ["call", "daughter", "tonight"], "cell phone",
     ["Can you help me call my daughter tonight?",
      "I want to call my daughter tonight.",
      "I'd like to phone my daughter tonight."]),
    ("Anything else?", ["read", "later"], "book",
     ["Could you read that book to me later?",
      "I'd like you to read that book to me later.",
      "I want to read that book later."]),
    ("How are you feeling?", ["head", "hurts", "little"], None,
     ["My head hurts a little.",
      "I have a little headache.",
      "My head is a little sore."]),
]

# Ask LM Studio to force valid JSON with exactly this shape (same approach as lamp-robot).
OPTIONS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "options",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "options": {"type": "array", "items": {"type": "string"},
                            "minItems": 3, "maxItems": 3},
            },
            "required": ["options"],
            "additionalProperties": False,
        },
    },
}

# Extra instruction added only when we re-ask after the filter dropped too many.
RETRY_NOTE = ("Your last options were spoken TO the patient. Write 3 sentences "
              "the PATIENT would say, in first person (I, me, my).")


def build_user_message(fragments, caregiver_question=None, pointing_at=None):
    """The text Bonsai sees for one turn. Examples use this same format."""
    lines = []
    if caregiver_question:
        lines.append(f'Caregiver just asked: "{caregiver_question}"')
    lines.append("Fragments: " + json.dumps(fragments))
    if pointing_at:
        # Later, a camera will fill this in. For now it's passed by hand (or None).
        lines.append(f"Patient is pointing at: {pointing_at}")
    return "\n".join(lines)


def parse_options(raw):
    """Returns (options, how). Tries JSON first; if that fails, splits on newlines."""
    try:
        data = json.loads(raw)
        options = [str(o).strip() for o in data["options"] if str(o).strip()]
        if options:
            return options[:3], "json"
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    # Fallback: one option per line. Strip list markers like "1." "-" and quotes.
    lines = []
    for line in raw.splitlines():
        line = re.sub(r'^\s*(\d+[.)]|[-*•])\s*', "", line).strip().strip('",')
        if line and not line.startswith(("{", "}", "[", "]")):
            lines.append(line)
    return lines[:3], "newline fallback"


# ---------- First-person filter (plain Python) ----------

# Goal: keep sentences the PATIENT would say, drop ones said TO the patient.
#   keep: "Could I have a blanket?"   "Can you help me call my daughter?"
#         "Could you change the channel?"  "Please turn it down."   (requests to the caregiver)
#   drop: "Do you want cold water?"   "Would you like a drink?"  "Is it your son?"
ASKS_PATIENT_STARTS = ("do you", "would you like", "are you")      # questions TO the patient
REQUEST_STARTS = ("could you", "would you", "can you", "please")   # patient asking caregiver
YOUR_WORDS = {"your", "yours", "yourself"}                         # "your" = the patient's
YOU_WORDS = {"you"}
FIRST_PERSON_WORDS = {"i", "me", "my", "mine", "myself"}
QUESTION_WORDS = {"who", "what", "where", "when", "why", "how"}


def words_in(text):
    """Lowercase words. Apostrophes split words, so "I'd" gives "i" and "d"."""
    return set(re.findall(r"[a-z]+", text.lower().replace("’", "'")))


def fragments_are_question(fragments):
    """True if the patient seems to be asking something, i.e. a fragment contains
    who/what/where/when/why/how. Whisper's "?" is ignored: it adds one whenever
    the voice goes up ("Want?"), which doesn't mean the patient asked anything."""
    return any(words_in(frag) & QUESTION_WORDS for frag in fragments)


def is_aimed_at_patient(option):
    """True if this option sounds like someone talking TO the patient.
    The checks run in order; the first one that matches decides."""
    text = option.strip().lower().replace("’", "'")
    words = words_in(text)
    # 1. Asks the patient something: "Do you want...", "Would you like...", "Are you..."
    if text.startswith(ASKS_PATIENT_STARTS):
        return True
    # 2. Says "your", which here means the patient's ("your son", "your wife").
    if words & YOUR_WORDS:
        return True
    # 3. A request the patient makes to the caregiver: "Could you...", "Please..."
    if text.startswith(REQUEST_STARTS):
        return False
    # 4. Mentions "you" but never "I/me/my": probably not the patient speaking.
    return bool(words & YOU_WORDS) and not (words & FIRST_PERSON_WORDS)


def first_person_filter(options, fragments):
    """Drop options aimed at the patient, unless the patient was asking a question."""
    if fragments_are_question(fragments):
        return list(options)
    return [o for o in options if not is_aimed_at_patient(o)]


class Brain:
    def __init__(self, base_url=LLM_BASE_URL, model=LLM_MODEL):
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        self.model = model

    def warm_up(self):
        """Send one tiny request so LM Studio loads the model before the first real turn."""
        self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )

    def _ask(self, user_message, extra_note=None):
        """One call to Bonsai. Returns (options, how_parsed)."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for question, frags, pointing, options in EXAMPLES:
            messages.append({"role": "user",
                             "content": build_user_message(frags, question, pointing)})
            messages.append({"role": "assistant", "content": json.dumps({"options": options})})
        if extra_note:
            user_message += "\n" + extra_note
        messages.append({"role": "user", "content": user_message})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format=OPTIONS_SCHEMA,
            temperature=0.2,
            max_tokens=200,
        )
        msg = resp.choices[0].message
        # Some LM Studio models put structured output in reasoning_content.
        raw = msg.content or getattr(msg, "reasoning_content", None) or ""
        return parse_options(raw)

    def suggest(self, fragments, caregiver_question=None, pointing_at=None):
        """Returns a dict with:
            options  - the filtered options to show (may be fewer than 3)
            how      - "json" or "newline fallback"
            dropped  - options the first-person filter removed
            retried  - True if we had to ask Bonsai a second time
        """
        user_message = build_user_message(fragments, caregiver_question, pointing_at)
        raw_options, how = self._ask(user_message)
        options = first_person_filter(raw_options, fragments)
        dropped = [o for o in raw_options if o not in options]
        retried = False

        # Fewer than 2 good options left? Ask once more, with a reminder.
        if len(options) < 2:
            retried = True
            raw2, how = self._ask(user_message, extra_note=RETRY_NOTE)
            kept2 = first_person_filter(raw2, fragments)
            dropped += [o for o in raw2 if o not in kept2]
            # Keep the good ones from both tries, no duplicates, at most 3.
            for o in kept2:
                if o not in options:
                    options.append(o)
            options = options[:3]

        return {"options": options, "how": how, "dropped": dropped, "retried": retried}
