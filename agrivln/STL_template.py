# AgriVLN - Subtask List Module, version 3: fixed template ("approach, then confirm")
#
# Why: with v8/v9 the 7B model kept copying the arrival wording into the NEAR condition, even after
# five corrections. So the structure is no longer left to the model. Only two pieces are taken from the
# instruction, and the rest is a fixed template that is the same for every episode:
#
#   movement  = what to do first            (the part of the instruction before "until"/"when")
#   target    = the final target's name      (the noun phrase right after "until"/"when")
#   arrival   = the instruction's own arrival wording (after "until"/"when"), unchanged
#
#   1. <movement>                          end: <target> is visible ahead (it may still be small and far)
#   2. Keep moving toward <target>.        end: <target> is the largest object in view and its details are clearly visible
#   3. Stop when <arrival>.                end: <arrival>, with "you" -> "the robot"
#
# Pieces are found with simple text rules; if a rule fails, the local LLM is asked for them (temperature 0)
# and its answer must be an exact substring of the instruction. Nothing is tuned to an episode.
# Side effect: the target name is extracted automatically (perception.py can use it).

import json
import re

_CUT = re.compile(r"\b(?:and\s+)?(?:keep\s+(?:moving|walking|going|driving)\s+)?(until|when)\b", re.I)


def split_instruction(instruction):
    """Return (movement, target, arrival) or None if the text rules do not apply."""
    text = instruction.strip().rstrip(".")
    m = None
    for m in _CUT.finditer(text):      # the last "until"/"when"
        pass
    if not m:
        return None
    movement = text[:m.start()].strip(" ,")
    movement = re.sub(r"\s+and\s+stop$", "", movement, flags=re.I).strip(" ,")
    arrival = text[m.end():].strip(" ,")
    arrival = re.sub(r",?\s*then\s+stop$", "", arrival, flags=re.I).strip(" ,")
    t = re.match(r"(.+?)\s+(?:is|are)\s+", arrival)
    if not movement or not t:
        return None
    return movement[0].upper() + movement[1:] + ".", t.group(1).strip(), arrival


def ask_llm(my_model, instruction):
    import ollama
    q = ("From this navigation instruction, copy three pieces EXACTLY as they are written (no new words):\n"
         "movement: what the robot does before the final target is reached\n"
         "target: the name of the final target object (a noun phrase starting with 'the')\n"
         "arrival: the words that describe where the robot must stop, starting with the target\n"
         'Answer as JSON: {"movement": "...", "target": "...", "arrival": "..."}\n\n'
         f"Instruction: {instruction}")
    r = ollama.chat(model=my_model, messages=[{"role": "user", "content": q}],
                    options={"temperature": 0, "seed": 42})
    m = re.search(r"\{.*\}", r["message"]["content"], re.S)
    d = json.loads(m.group(0)) if m else {}
    low = instruction.lower()
    if all(isinstance(d.get(k), str) and d[k] and d[k].lower().rstrip(".") in low for k in ("movement", "target", "arrival")):
        mv = d["movement"].strip(" ,.")
        return mv[0].upper() + mv[1:] + ".", d["target"].strip(), d["arrival"].strip(" ,.")
    return None


def _robot(s):
    s = re.sub(r"\byou\b", "the robot", s)
    return re.sub(r"\byour\b", "the robot's", s)


def build_stl(instruction, my_model="qwen2.5vl:7b"):
    parts = split_instruction(instruction)
    source = "text rules"
    if parts is None:
        parts, source = ask_llm(my_model, instruction), "LLM (exact copy)"
    if parts is None:
        raise ValueError("could not find movement / target / arrival in the instruction")
    movement, target, arrival = parts
    visible = f"{target} is visible ahead (it may still be small and far)"
    near = f"{target} is the largest object in view and its details are clearly visible"
    stl = [
        {"step": 1, "subtask": movement, "start_condition": "always", "end_condition": visible},
        {"step": 2, "subtask": f"Keep moving toward {target}.", "start_condition": visible, "end_condition": near},
        {"step": 3, "subtask": f"Stop when {arrival}.", "start_condition": near, "end_condition": _robot(arrival)},
    ]
    return stl, {"source": source, "movement": movement, "target": target, "arrival": arrival}
