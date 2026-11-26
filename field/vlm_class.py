# vlm_class.py  -  "WHAT is on the path?" one-word class from the VLM (used by vlm_select.py and field_client.py --policy fusion)
import re, time

CLASSES = ["PERSON", "ANIMAL", "VEHICLE", "OBJECT", "GRASS", "HOLE", "NOTHING"]

WORDINGS = {
    "W1": ("This image comes from the front camera of a small farm robot driving along a dirt path between crop beds.\n"
           "What is on the path directly ahead of the robot, closest to it?\n"
           "Answer with one word from this list: PERSON, ANIMAL, VEHICLE, OBJECT, GRASS, HOLE, NOTHING."),
    "W2": ("This image comes from the front camera of a small farm robot driving along a dirt path between crop beds.\n"
           "Look only at the dirt path straight ahead of the robot. Choose the one word that best describes what is ON or IN the path:\n"
           "PERSON (a human, standing, sitting or lying), ANIMAL, VEHICLE, OBJECT (rock, drum, box, pipe), "
           "GRASS (plants the robot can drive through), HOLE (ditch, gap or step down), NOTHING (the path is clear).\n"
           "Answer with one word."),
}

# what the class means for the decision
STOP_CLASSES = ("PERSON", "ANIMAL", "HOLE")      # never detour
DETOUR_CLASSES = ("OBJECT", "VEHICLE")           # detour if there is side space
PASS_CLASSES = ("GRASS",)                         # drive through (if depth agrees it is low)


def parse(text):
    found = re.findall(r"\b(" + "|".join(CLASSES) + r")\b", (text or "").upper())
    return found[-1] if found else "?"


def ask(model, wording, jpg_bytes):
    import ollama
    t0 = time.time()
    r = ollama.chat(model=model, messages=[{"role": "user", "content": WORDINGS[wording], "images": [jpg_bytes]}],
                    options={"temperature": 0, "seed": 42})
    return parse(r["message"]["content"]), time.time() - t0
