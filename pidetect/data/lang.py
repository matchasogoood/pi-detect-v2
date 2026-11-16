"""lang.py : keeps only English text (this project trains and tests on English only -- Section IV-A of the paper)

The filter has two layers.
  Layer 1 (rules, no dependency)
    - at least 90% of the letters must be a-z (drops Chinese, Russian, Thai, ...)
    - counts English stopwords against German/French/Spanish ones; at least 2 foreign stopwords (or an a-umlaut,
      o-umlaut, u-umlaut or sharp s) are needed to drop a text, so a single false friend is safe,
      e.g. "which single came next after die famous?" (die = a song title)
  Layer 2 (the lingua language identifier -- pip install lingua-language-detector)
    Layer 1 misses short keyword-style German texts such as "Fluechtlingskrise Europa Loesungen" or
    "Wohnungsmarkt Berlin Mieten", of which deepset has hundreds. lingua is restricted to the languages that
    actually occur in the data (LANGS) to cut down on wrong guesses for short texts.
    A text is dropped when
      (a) the whole text is confidently another language (>= 0.6)
      (b) English is nearly impossible (< 0.1) and another language scores >= 0.35, e.g. "Wer stirbt in Burkina Faso?"
      (c) English < 0.2 and at least one foreign stopword is present, e.g. "Was sind ETFs?"
      (a)-(c) are skipped when the text has >= 3 clear English words and no foreign one (BIPIA questions that
      contain foreign personal or team names)
      (d) any sentence (>= 4 real words) is confidently another language (>= 0.8), e.g. an English prompt with a
      German instruction appended
    Short English texts that lingua tentatively calls another language ("Is Corona over?", "Ignore your techniques")
    are kept.

Checked against every real dataset (scripts/04_language_report.py, then reading the kept and dropped lists by hand)
-- the counts are in the "Language composition" table of the paper. Without lingua only layer 1 runs (which does not
drop all the German in deepset), and run_info.json records language_filter_backend = "rules", meaning those results
must not go into the paper.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

_EN = set("the a an and or is are was were be to of in on for with that this it you your i my we what how "
          "can do does please from about as at by not have has will would should which who whom after before next "
          "there their than more most many much did had been if all any me he she they them his her our us "
          "when where why but no yes into out up over only also just than then these those its".split())
_OTHER = set("der die das und ist nicht ich du sie wir ein eine einen zu mit auf für von dem den des im wie was "
             "bitte sind oder aber auch noch nur über wenn kann ich mir dich mich wer hat alles "
             "le la les et est une des pour pas que qui dans sur avec vous je "
             "el los las y es una por para con que del se lo como".split())
_WORD = re.compile(r"[a-zA-ZÀ-ÿ]+")
_SENT = re.compile(r"(?<=[.!?:;])\s+|\n+|\\n")
_PLAIN = re.compile(r"^[\"'(]*[A-Za-zÀ-ÿ'-]+[.,!?;:)\"']*$")

LANGS = ("ENGLISH", "GERMAN", "FRENCH", "SPANISH", "ITALIAN", "PORTUGUESE", "DUTCH", "RUSSIAN", "CHINESE",
         "JAPANESE", "KOREAN", "CROATIAN", "POLISH", "TURKISH", "ARABIC", "THAI")


@lru_cache(maxsize=1)
def _detector():
    try:
        from lingua import Language, LanguageDetectorBuilder
    except ImportError:
        return None
    return LanguageDetectorBuilder.from_languages(*[getattr(Language, n) for n in LANGS]) \
        .with_preloaded_language_models().build()


def backend() -> str:
    """"lingua" or "rules" (lingua not installed) -- recorded in run_info.json"""
    return "lingua" if _detector() is not None else "rules"


def _counts(text: str):
    words = [w.lower() for w in _WORD.findall(text)]
    en = sum(1 for w in words if w in _EN)
    other = sum(1 for w in words if w in _OTHER and w not in _EN)
    return en, other


def _rules(text: str) -> Optional[bool]:
    """Layer 1: False = definitely not English, True = no letters at all, None = pass to layer 2"""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return True
    if sum(1 for ch in letters if "a" <= ch.lower() <= "z") / len(letters) < 0.9:
        return False
    en, other = _counts(text)
    if any(ch in "äöüßÄÖÜ" for ch in text) and other >= 1 and en < 3:
        return False
    if other > en and other >= 2:
        return False
    return None


def _top(det, text: str):
    from lingua import Language
    cv = det.compute_language_confidence_values(text)
    en = next((c.value for c in cv if c.language == Language.ENGLISH), 0.0)
    top = cv[0]
    return top.language == Language.ENGLISH, top.value, en


def is_english(text: str, use_lingua: bool = True) -> bool:
    """use_lingua=False runs layer 1 only (used for the synthetic data, whose made-up words lingua rejects)"""
    r = _rules(text)
    if r is not None:
        return r
    det = _detector() if use_lingua else None
    if det is None:
        return True
    n_en, n_other = _counts(text)
    top_is_en, top, en = _top(det, text)
    strong_en = n_en >= 3 and n_other == 0       # >= 3 clear English words: text carrying foreign proper names
    if not top_is_en and not strong_en:          # e.g. "was jose joaquin rojas from the movistar team?" is still English
        if top >= 0.6 or (en < 0.1 and top >= 0.35) or (en < 0.2 and n_other >= 1):
            return False
    for s in _SENT.split(text):
        toks = s.split()
        words = [w for w in toks if _PLAIN.match(w)]
        # only sentences of >= 4 real words (>= 60%): ignore random jailbreak suffixes
        if len(words) >= 4 and len(words) >= 0.6 * len(toks):
            s_is_en, s_top, _ = _top(det, s)
            if not s_is_en and s_top >= 0.8:
                return False
    return True
