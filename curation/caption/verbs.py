"""27-class action semantics from the MMFi dataset annotation.

Source: MMFi_dataset README (Activity A01-A27). Each phrase is the natural
language verb phrase a human would use to describe the action. These anchor
the synthetic captions and the contrastive text side.
"""
from __future__ import annotations

# (code -> natural-language verb phrase)
ACTION_PHRASES = {
    "A01": "stretching and relaxing",
    "A02": "expanding chest horizontally",
    "A03": "expanding chest vertically",
    "A04": "twisting left",
    "A05": "twisting right",
    "A06": "marking time in place",
    "A07": "extending the left limb",
    "A08": "extending the right limb",
    "A09": "lunging toward the left front",
    "A10": "lunging toward the right front",
    "A11": "extending both limbs",
    "A12": "squatting down",
    "A13": "raising the left hand",
    "A14": "raising the right hand",
    "A15": "lunging to the left side",
    "A16": "lunging to the right side",
    "A17": "waving the left hand",
    "A18": "waving the right hand",
    "A19": "picking up things",
    "A20": "throwing toward the left side",
    "A21": "throwing toward the right side",
    "A22": "kicking toward the left side",
    "A23": "kicking toward the right side",
    "A24": "extending the left side of the body",
    "A25": "extending the right side of the body",
    "A26": "jumping up",
    "A27": "bowing",
}


def action_code(label: int) -> str:
    """label (0-26) -> MMFi action code (A01-A27)."""
    return f"A{label + 1:02d}"


def LABEL_TO_VERB(label: int) -> str:
    """label (0-26) -> natural-language verb phrase."""
    return ACTION_PHRASES[action_code(label)]
