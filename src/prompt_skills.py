"""Map free-text prompts onto pick / place / push / reach."""

from __future__ import annotations

SKILLS = ("pick", "place", "push", "reach")

_PLACE = ("place", "put", "drop", "set down", "onto the plate", "on the plate", "on plate")
_PUSH = ("push", "slide", "shove", "nudge")
_PICK = ("pick", "grasp", "grab", "lift", "take")
_REACH = ("reach", "go to", "move to", "touch")

TEMPLATES = {
    "pick": (
        "pick the red cube",
        "grasp the cube",
        "lift the cube",
        "pick up the block",
    ),
    "place": (
        "put the cube on the plate",
        "place the cube onto the plate",
        "set the block on the plate",
        "put the red cube on the plate",
    ),
    "push": (
        "push the cube to the side",
        "slide the block sideways",
        "push the red cube",
        "nudge the cube away",
    ),
    "reach": (
        "reach the cube",
        "move to the cube",
        "go to the red cube",
        "touch the block",
    ),
}


def parse_skill(prompt: str) -> str:
    text = (prompt or "").strip().lower()
    if not text:
        return "pick"
    if any(k in text for k in _PLACE):
        return "place"
    if any(k in text for k in _PUSH):
        return "push"
    if any(k in text for k in _REACH) and not any(k in text for k in _PICK):
        return "reach"
    if any(k in text for k in _PICK):
        return "pick"
    return "pick"


def canonical_instruction(prompt: str) -> str:
    skill = parse_skill(prompt)
    return TEMPLATES[skill][0]


def sample_instruction(skill: str, rng=None) -> str:
    import numpy as np

    rng = np.random.default_rng() if rng is None else rng
    opts = TEMPLATES[skill]
    return str(opts[int(rng.integers(0, len(opts)))])
