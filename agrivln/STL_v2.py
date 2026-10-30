# AgriVLN - Subtask List Module, version 2 ("approach, then confirm"), revision v8
#
# Same job as agrivln/STL.py (instruction -> subtask list, same JSON format, so decide.py
# and decide_quantum.py work unchanged). The prompt enforces:
#   1. approach first, then confirm: "visible" is never "arrived"
#   2. the FINAL end_condition keeps the instruction's own arrival wording (the instruction
#      defines arrival; closeness cues may be added, never substituted)
#   3. NEAR is described with comparisons the VLM can check (largest object, beside instead of
#      ahead, no free ground in front), never numbers or fractions of the image
#   4. no left/right that the instruction does not give for the target
#   5. only the final subtask stops; no repeated conditions
# v8 changes (after reading the v7 lists, NOT after looking at scores):
#   - two examples (target ahead, target beside) so one example's side is not copied
#   - rules 2-4 are new, plus automatic checks for them in check()
# v9 (after reading the v8 lists): NEAR must not reuse the arrival wording; the closeness cue must
#   match ahead/beside. Both are checked in check().
# Generation: temperature 0, fixed seed, checked, retried if broken.

import json
import re

import ollama

SYSTEM = """
You are an expert in Vision-and-Language Navigation (VLN) for an agricultural robot with a low, forward-facing,
wide-angle camera. You turn one navigation instruction into a subtask list that the robot checks against its
camera image every 0.2 seconds. A small vision-language model will read your conditions, so they must be easy
to check by looking at one image.

Fields of each subtask:
- "step": order number (1, 2, 3, ...)
- "subtask": what the robot does in this phase
- "start_condition": "always" for step 1, otherwise exactly the previous end_condition
- "end_condition": what the robot can SEE in one camera image when the phase is finished

Rules:
1. Approach first, then confirm. Use exactly 3 subtasks:
   step 1 ends when the final target first becomes visible (it is usually still far away);
   step 2 ends when the target is NEAR;
   step 3 is the final approach and ends when the robot has ARRIVED.
2. ARRIVED (the final end_condition) must keep the arrival wording of the instruction itself, nearly word for word.
   You may add ONE closeness cue after it, joined with "and", and it must fit where the target ends up:
   target AHEAD (in front)  -> "and no free ground is visible between the robot and the <target>"
   target BESIDE (next to)  -> "and the <target> is no longer ahead of the robot"
   Never replace the instruction's wording, and never exaggerate it (do not write "fills the entire image" if the
   instruction does not say so).
3. NEAR (step 2's end_condition) is a comparison that can be checked by eye, never a number, and it must NOT reuse
   the instruction's arrival wording (near comes BEFORE arrived, so it must be a weaker, earlier condition):
   GOOD: "the <target> is the largest object in view", "the <target> is close and its details are clearly visible",
         "the <target> is beside the path instead of far ahead"
   BAD:  "half the image width", "a quarter of the image height", "more than 2 metres", "50 %".
4. Do not mention left or right for the target unless the instruction gives that side for the target.
5. Name the target with its distinguishing features (shape, material, colour, attached parts) so a similar-looking
   object is not mistaken for it.
6. Only the final subtask involves stopping: its "subtask" starts with "Stop when" and repeats the instruction's
   arrival wording. The word "stop" must not appear in any other subtask.
7. Every end_condition must be different. One short sentence each, about the image only.

Example A (target ends up AHEAD; different scene, format only):
<instruction> Drive down the orchard track and stop when the wooden gate is right in front of you. </instruction>
<thought> The gate can be seen from far down the track, so seeing it is not arriving. Near = the gate is the largest
object in view. Arrived = the instruction's wording plus a closeness cue for a target ahead. No side is given. </thought>
<subtask_list>
[
   {"step": 1, "subtask": "Drive forward along the orchard track.", "start_condition": "always",
    "end_condition": "a wooden gate is visible ahead on the track (it may still be small and far)"},
   {"step": 2, "subtask": "Keep driving toward the wooden gate.", "start_condition": "a wooden gate is visible ahead on the track (it may still be small and far)",
    "end_condition": "the wooden gate is the largest object in view and its planks are clearly visible"},
   {"step": 3, "subtask": "Stop when the wooden gate is right in front of you.", "start_condition": "the wooden gate is the largest object in view and its planks are clearly visible",
    "end_condition": "the wooden gate is right in front of the robot and no free ground is visible between the robot and the gate"}
]
</subtask_list>

Example B (target ends up BESIDE; different scene, format only):
<instruction> Follow the greenhouse row and stop when the blue water tank is right next to you on your right. </instruction>
<thought> The instruction gives the side (right), so I may use it. Near = the tank is close, beside the row instead of
far ahead. Arrived = the instruction's wording plus a closeness cue for a target beside. </thought>
<subtask_list>
[
   {"step": 1, "subtask": "Move forward along the greenhouse row.", "start_condition": "always",
    "end_condition": "a blue cylindrical water tank is visible ahead (it may still be small and far)"},
   {"step": 2, "subtask": "Keep moving forward toward the blue water tank.", "start_condition": "a blue cylindrical water tank is visible ahead (it may still be small and far)",
    "end_condition": "the blue water tank is close, beside the row instead of far ahead"},
   {"step": 3, "subtask": "Stop when the blue water tank is right next to you on your right.", "start_condition": "the blue water tank is close, beside the row instead of far ahead",
    "end_condition": "the blue water tank is right next to the robot on its right and is no longer ahead of the robot"}
]
</subtask_list>

Output format:
<thought> {where the target first becomes visible, what NEAR looks like, the instruction's ARRIVED wording} </thought>
<subtask_list>
[ ... ]
</subtask_list>
"""

_STOP_WORDS = set("a an the and or of to is are be you your it its in on at with then stop when until "
                  "keep moving move walk go along".split())


def arrival_clause(instruction):
    """Text after the last 'until' / 'when' in the instruction (where arrival is described)."""
    low = instruction.lower()
    cut = max(low.rfind(" until "), low.rfind(" when "))
    clause = low[cut:] if cut >= 0 else low
    return re.sub(r",?\s*then stop\.?$", "", clause.strip())


def _words(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP_WORDS]


def check(stl, instruction=None):
    """Return a list of problems (empty = valid)."""
    problems = []
    if not isinstance(stl, list) or len(stl) < 3:
        return ["need at least 3 subtasks"]
    keys = {"step", "subtask", "start_condition", "end_condition"}
    for i, s in enumerate(stl):
        if not keys <= set(s):
            return [f"subtask {i + 1} misses fields"]
        if s["step"] != i + 1:
            problems.append("step numbers not 1..n")
        if i < len(stl) - 1 and "stop" in s["subtask"].lower():
            problems.append(f"'stop' in non-final subtask {i + 1}")
    if str(stl[0]["start_condition"]).strip().lower() != "always":
        problems.append("first start_condition is not 'always'")
    if not stl[-1]["subtask"].strip().lower().startswith("stop when"):
        problems.append("final subtask does not start with 'Stop when'")
    ends = [s["end_condition"].strip().lower() for s in stl]
    if len(set(ends)) < len(ends):
        problems.append("repeated end_condition")
    if re.sub(r"[^a-z ]", "", ends[-1]).strip() in ("stop", "stopped", "robot stops", "the robot stops"):
        problems.append("final end_condition is just 'stop', not something visible")
    if ends[-1] == stl[-1]["start_condition"].strip().lower():
        problems.append("final end_condition equals its start_condition (near = arrived)")
    if re.fullmatch(r".*\b(is |are )?(visible|in view|ahead|seen)\.?", ends[-1]):
        problems.append("final end_condition is only 'visible/ahead'")

    # v8 checks on the NEAR and ARRIVED conditions
    for i in range(1, len(stl)):
        c = ends[i]
        if re.search(r"\d|%|\bat least\b|\bmore than\b|\bless than\b|\bpercent\b|image (width|height)", c):
            problems.append(f"step {i + 1} end_condition uses a number or measure: '{c}'")
    if instruction:
        arr = arrival_clause(instruction)
        for i in range(1, len(stl)):
            for side in ("left", "right"):
                if re.search(rf"\b{side}\b", ends[i]) and not re.search(rf"\b{side}\b", arr):
                    problems.append(f"step {i + 1} end_condition adds '{side}', not given for the target")
        for word in ("entire", "whole", "completely"):
            if re.search(rf"\b{word}\b", ends[-1]) and word not in instruction.lower():
                problems.append(f"final end_condition exaggerates ('{word}')")
        target = set(_words(ends[0]))           # words naming the target (from "X is visible ahead")
        extra = [w for w in _words(arr) if w not in target]
        near = set(_words(ends[1]))
        if extra and sum(w in near for w in extra) / len(extra) > 0.5:
            problems.append("step 2 (near) reuses the instruction's arrival wording; near must come before arrived")
        beside = re.search(r"\b(beside|next to|alongside)\b", arr)
        if beside and "free ground" in ends[-1]:
            problems.append("target ends up beside, but the final cue 'no free ground' is for a target ahead")
        if not beside and "no longer ahead" in ends[-1]:
            problems.append("target ends up ahead, but the final cue 'no longer ahead' is for a target beside")
        need = _words(arr)
        have = set(_words(ends[-1]))
        if need:
            share = sum(w in have for w in need) / len(need)
            if share < 0.6:
                problems.append(f"final end_condition keeps only {share:.0%} of the instruction's arrival words")
    return problems


def generate_stl_v2(my_model, instruction, seed=42, tries=5):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"<instruction> {instruction} </instruction>"}]
    last = None
    for k in range(tries):
        r = ollama.chat(model=my_model, messages=messages, options={"temperature": 0, "seed": seed + k})
        text = r["message"]["content"]
        m = re.search(r"<subtask_list>\s*(\[.*?\])\s*</subtask_list>", text, re.DOTALL)
        try:
            stl = json.loads(m.group(1)) if m else None
        except json.JSONDecodeError:
            stl = None
        problems = check(stl, instruction) if stl else ["no valid JSON in <subtask_list>"]
        print(f"[STL2] try {k + 1} (seed {seed + k}): " + ("OK" if not problems else "; ".join(problems)))
        if not problems:
            return stl, text
        last = (stl, text)
        # tell the model what was wrong and ask again
        messages = messages[:2] + [{"role": "assistant", "content": text},
                                   {"role": "user", "content": "Please fix these problems and output the full list again: "
                                    + "; ".join(problems)}]
    print("[STL2][WARNING] no valid list after all tries; returning the last one (check it by hand)")
    return last
