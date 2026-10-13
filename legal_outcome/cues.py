"""Configurable, case-insensitive legal cue counts. Counts are not legal rules."""
import re
import torch

# The source gives K=12 but does not name the cues. These are explicit defaults.
DEFAULT_CUES = {
    "statute": r"\b(?:section|article|statute|act)\b",
    "precedent": r"\b(?:precedent|held|ruling|cited)\b",
    "evidence": r"\b(?:evidence|exhibit|testimony|witness)\b",
    "burden": r"\b(?:burden of proof|beyond reasonable doubt|balance of probabilities)\b",
    "negation": r"\b(?:not|no|never|without)\b",
    "grant": r"\b(?:allow(?:ed)?|grant(?:ed)?|uphold|affirm(?:ed)?)\b",
    "deny": r"\b(?:dismiss(?:ed)?|deny|denied|reject(?:ed)?)\b",
    "appeal": r"\b(?:appeal|appellant|respondent)\b",
    "liability": r"\b(?:liable|liability|guilty|convict(?:ed|ion)?)\b",
    "remedy": r"\b(?:damages|compensation|injunction|relief)\b",
    "procedure": r"\b(?:jurisdiction|limitation|admissib(?:le|ility)|procedure)\b",
    "obligation": r"\b(?:shall|must|duty|obligation)\b",
}


class CueExtractor:
    def __init__(self, patterns=None):
        patterns = DEFAULT_CUES if patterns is None else patterns
        if len(patterns) != 12:
            raise ValueError("The architecture requires exactly 12 cue features")
        self.patterns = dict(patterns)
        self.compiled = [re.compile(p, re.IGNORECASE) for p in patterns.values()]

    def __call__(self, spans):
        return torch.tensor([[len(p.findall(s)) for p in self.compiled] for s in spans], dtype=torch.float32)
