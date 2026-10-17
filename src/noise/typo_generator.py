"""
Controlled Typographical Noise Generator for NLP Robustness Research.

Simulates realistic human typing errors on standard QWERTY keyboards:
1. Keyboard Adjacency Substitution: Replacing a character with a physically adjacent key.
2. Character Deletion: Omission of a character in words of length > 3.
3. Character Insertion: Adding an adjacent QWERTY key.
4. Adjacent Transposition: Swapping adjacent characters in words of length >= 2.

Preserves URLs, user mentions (@username), and whitespace structures.
Operates deterministically using an isolated random.Random instance.
"""

import random
from typing import Dict, List, Optional


# Standard QWERTY keyboard physical adjacency map (lowercase)
QWERTY_ADJACENCY: Dict[str, List[str]] = {
    'q': ['w', 'a', 's'],
    'w': ['q', 'e', 'a', 's', 'd'],
    'e': ['w', 'r', 's', 'd', 'f'],
    'r': ['e', 't', 'd', 'f', 'g'],
    't': ['r', 'y', 'f', 'g', 'h'],
    'y': ['t', 'u', 'g', 'h', 'j'],
    'u': ['y', 'i', 'h', 'j', 'k'],
    'i': ['u', 'o', 'j', 'k', 'l'],
    'o': ['i', 'p', 'k', 'l'],
    'p': ['o', 'l'],
    'a': ['q', 'w', 's', 'z'],
    's': ['a', 'w', 'e', 'd', 'x', 'z'],
    'd': ['s', 'e', 'r', 'f', 'c', 'x'],
    'f': ['d', 'r', 't', 'g', 'v', 'c'],
    'g': ['f', 't', 'y', 'h', 'b', 'v'],
    'h': ['g', 'y', 'u', 'j', 'n', 'b'],
    'j': ['h', 'u', 'i', 'k', 'm', 'n'],
    'k': ['j', 'i', 'o', 'l', 'm'],
    'l': ['k', 'o', 'p'],
    'z': ['a', 's', 'x'],
    'x': ['z', 's', 'd', 'c'],
    'c': ['x', 'd', 'f', 'v'],
    'v': ['c', 'f', 'g', 'b'],
    'b': ['v', 'g', 'h', 'n'],
    'n': ['b', 'h', 'j', 'm'],
    'm': ['n', 'j', 'k']
}


class TypoGenerator:
    """
    Generates controlled typographical noise at specified severity rates.
    """

    def __init__(self, seed: int = 42):
        """
        Initializes the typo generator with a deterministic seed.

        Args:
            seed: Integer random seed for reproducibility.
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

    def _get_adjacent_key(self, char: str) -> str:
        """
        Returns a randomly chosen adjacent key on the QWERTY keyboard,
        preserving case if upper.
        """
        is_upper = char.isupper()
        lower_char = char.lower()
        neighbors = QWERTY_ADJACENCY.get(lower_char)
        if not neighbors:
            return char
        chosen = self.rng.choice(neighbors)
        return chosen.upper() if is_upper else chosen

    def perturb_word(self, word: str, severity: float) -> str:
        """
        Perturbs a single word token based on character-level probability.

        Args:
            word: Input word token.
            severity: Probability (0.0 to 1.0) of perturbing each character.

        Returns:
            Perturbed word string.
        """
        if severity <= 0.0 or not word:
            return word

        # Preserve mentions, URLs, hashtags
        if word.startswith(("@", "http://", "https://")):
            return word

        chars = list(word)
        n = len(chars)
        if n == 0:
            return word

        perturbed_chars = []
        i = 0
        while i < n:
            ch = chars[i]
            if not ch.isalpha():
                perturbed_chars.append(ch)
                i += 1
                continue

            if self.rng.random() < severity:
                # Select one of the 4 typo operations:
                # 0: QWERTY substitution
                # 1: Character deletion (only if current length > 3)
                # 2: Character insertion (adjacent key)
                # 3: Adjacent transposition (if remaining chars >= 2)
                ops = [0, 2]
                if n > 3 and len(perturbed_chars) + (n - i) > 2:
                    ops.append(1)
                if i + 1 < n and chars[i + 1].isalpha():
                    ops.append(3)

                chosen_op = self.rng.choice(ops)

                if chosen_op == 0:
                    # QWERTY substitution
                    perturbed_chars.append(self._get_adjacent_key(ch))
                    i += 1
                elif chosen_op == 1:
                    # Deletion: drop this character
                    i += 1
                elif chosen_op == 2:
                    # Insertion: insert an adjacent character alongside current
                    adj = self._get_adjacent_key(ch)
                    if self.rng.random() < 0.5:
                        perturbed_chars.extend([adj, ch])
                    else:
                        perturbed_chars.extend([ch, adj])
                    i += 1
                elif chosen_op == 3:
                    # Transposition: swap chars[i] and chars[i + 1]
                    perturbed_chars.append(chars[i + 1])
                    perturbed_chars.append(ch)
                    i += 2
            else:
                perturbed_chars.append(ch)
                i += 1

        result = "".join(perturbed_chars)
        # Fallback if somehow word became empty
        return result if result else word

    def perturb_text(self, text: str, severity: float) -> str:
        """
        Applies typographical noise to an entire sentence / document.

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
