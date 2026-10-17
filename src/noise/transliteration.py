"""
Hinglish Transliteration Variation Generator for NLP Robustness Research.

Simulates authentic orthographic and phonetic variations in Romanized Hindi (Hinglish):
1. Lexical / Whole-Word Transliteration Variations (e.g. 'nahi' <-> 'nhi' <-> 'nahin',
   'bhi' <-> 'b', 'bahut' <-> 'bohot' <-> 'bht', 'theek' <-> 'thik').
2. Vowel Alternations (e.g. 'aa' <-> 'a', 'ee' <-> 'i', 'oo' <-> 'u', 'ai' <-> 'e').
3. Consonant Digraph & Aspiration Variations (e.g. 'kh' <-> 'k', 'ch' <-> 'chh',
   'bh' <-> 'b', 'ph' <-> 'f', 'sh' <-> 's', 'z' <-> 'j', 'w' <-> 'v').

Preserves URLs, user mentions (@username), and capitalization patterns.
Operates deterministically using an isolated random.Random instance.
"""

import random
import re
from typing import Dict, List, Optional, Tuple


# Lexicon of frequent Hinglish words with authentic transliteration alternates
HINGLISH_LEXICON_VARIANTS: Dict[str, List[str]] = {
    "nahi": ["nhi", "nahin", "nae"],
    "nhi": ["nahi", "nahin"],
    "nahin": ["nahi", "nhi"],
    "bhi": ["b", "bhee", "bi"],
    "hai": ["h", "hae", "hay"],
    "hain": ["hn", "hai", "hay"],
    "acha": ["achha", "accha", "achhaa"],
    "achha": ["acha", "accha"],
    "accha": ["acha", "achha"],
    "kya": ["kyaa", "kyaaa", "kia"],
    "kyun": ["kyu", "kyo", "kyon"],
    "kyu": ["kyun", "kyo"],
    "lekin": ["lekn", "lekeen"],
    "aur": ["or", "awr"],
    "or": ["aur"],
    "ye": ["yeh", "ei"],
    "yeh": ["ye", "yeeh"],
    "wo": ["woh", "voh"],
    "woh": ["wo", "voh"],
    "bahut": ["bht", "bohot", "bahoot"],
    "bohot": ["bahut", "bht"],
    "bht": ["bahut", "bohot"],
    "kuch": ["kuchh", "kch"],
    "kuchh": ["kuch", "kch"],
    "theek": ["thik", "teek", "thek"],
    "thik": ["theek", "teek"],
    "samajh": ["samaj", "samjh"],
    "samaj": ["samajh"],
    "dekh": ["dek", "dekhh"],
    "dekho": ["deko", "dekhoo"],
    "yaar": ["yar", "yaaar"],
    "yar": ["yaar"],
    "bhai": ["bhaai", "bhaii", "bai"],
    "waqt": ["vaqt", "vakt", "wakt"],
    "zaroor": ["jaroor", "zarur"],
    "zyada": ["jyada", "jada"],
    "jyada": ["zyada", "jada"],
    "pehle": ["pahle", "pehele"],
    "kaise": ["kese", "kaisey"],
    "aise": ["ese", "aisey"],
    "koi": ["koyi", "koy"],
    "hoga": ["hogaa", "hoega"],
    "gaya": ["gya", "geya"],
    "raha": ["rha", "rahaa"],
    "rahi": ["rhi", "rahii"],
    "rahe": ["rhe", "rahee"],
    "wala": ["vala", "waala"],
    "wali": ["vali", "waali"],
    "wale": ["vale", "waale"],
    "dost": ["st", "dostt"],
    "sirf": ["srf", "shirf"],
    "pagal": ["paagal", "pgl"],
    "sahi": ["shai", "sahii"]
}

# Sub-word phonetic & digraph mappings (source -> list of targets)
PHONETIC_DIGRAPH_RULES: List[Tuple[str, List[str]]] = [
    ("chh", ["ch"]),
    ("ch", ["chh"]),
    ("kh", ["k"]),
    ("gh", ["g"]),
    ("jh", ["j"]),
    ("th", ["t"]),
    ("dh", ["d"]),
    ("bh", ["b"]),
    ("ph", ["f"]),
    ("sh", ["s"]),
    ("aa", ["a"]),
    ("ee", ["i"]),
    ("oo", ["u"]),
    ("ai", ["e"]),
    ("au", ["o"])
]


class TransliterationGenerator:
    """
    Generates Hinglish transliteration variation at specified severity rates.
    """

    def __init__(self, seed: int = 42):
        """
        Initializes the generator with a deterministic seed.

        Args:
            seed: Integer random seed.
        """
        self.seed = seed
        self.rng = random.Random(seed)

    def reset_seed(self, seed: Optional[int] = None) -> None:
        """
        Resets the internal PRNG state.
        """
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)

    def _match_case(self, original: str, replacement: str) -> str:
        """
        Applies original casing (all-caps, title-case, lower) to replacement.
        """
        if original.isupper():
            return replacement.upper()
        if original.istitle():
            return replacement.capitalize()
        return replacement.lower()

    def perturb_word(self, word: str, severity: float) -> str:
        """
        Perturbs a word token using Hinglish transliteration heuristics.

        Args:
            word: Input token.
            severity: Probability (0.0 to 1.0) of perturbing an eligible token.

        Returns:
            Perturbed word token.
        """
        if severity <= 0.0 or not word:
            return word

        # Preserve mentions, URLs, hashtags
        if word.startswith(("@", "http://", "https://")):
            return word

        # Strip surrounding punctuation to match lexical roots
        prefix_match = re.match(r"^(\W*)(.*?)(\W*)$", word)
        if not prefix_match:
            return word

        prefix, core, suffix = prefix_match.groups()
        if not core or not core.isalpha():
            return word

        core_lower = core.lower()

        # Step 1: Lexical lookup
        if core_lower in HINGLISH_LEXICON_VARIANTS:
            if self.rng.random() < severity:
                candidates = HINGLISH_LEXICON_VARIANTS[core_lower]
                chosen = self.rng.choice(candidates)
                new_core = self._match_case(core, chosen)
                return f"{prefix}{new_core}{suffix}"

        # Step 2: Sub-word phonetic digraph rules
        if self.rng.random() < severity:
            modified = core_lower
            applicable_rules = [
                (src, targets) for src, targets in PHONETIC_DIGRAPH_RULES
                if src in modified
            ]
            if applicable_rules:
                # Apply one chosen rule
                src, targets = self.rng.choice(applicable_rules)
                replacement = self.rng.choice(targets)
                # Replace only first occurrence to retain naturalism
                modified = modified.replace(src, replacement, 1)
                new_core = self._match_case(core, modified)
                return f"{prefix}{new_core}{suffix}"

        return word

    def perturb_text(self, text: str, severity: float) -> str:
        """
        Applies Hinglish transliteration variation to an entire sentence / document.

        Args:
            text: Input text string.
            severity: Perturbation severity rate (0.05, 0.10, 0.15, 0.20, etc.).

        Returns:
            Perturbed text string preserving token boundaries.
        """
        if severity <= 0.0 or not text:
            return text

        tokens = text.split(" ")
        perturbed_tokens = [self.perturb_word(tok, severity) for tok in tokens]
        return " ".join(perturbed_tokens)
