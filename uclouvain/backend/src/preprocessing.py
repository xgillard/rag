"""Preprocessing the documents to make them rag ready."""

import pathlib
import re
import unicodedata

REPLACEMENTS = str.maketrans(
    {
        unicodedata.lookup("LATIN CAPITAL LETTER AE"): "AE",
        unicodedata.lookup("LATIN SMALL LETTER AE"): "ae",
        unicodedata.lookup("LATIN CAPITAL LIGATURE OE"): "OE",
        unicodedata.lookup("LATIN SMALL LIGATURE OE"): "oe",
        unicodedata.lookup("TRADE MARK SIGN"): "(TM)",
        unicodedata.lookup("PERCENT SIGN"): "%",
        unicodedata.lookup("PER MILLE SIGN"): "<pour mille>",
        unicodedata.lookup("LATIN SMALL LETTER F WITH HOOK"): "f",
        unicodedata.lookup("MODIFIER LETTER CIRCUMFLEX ACCENT"): "^",
        # quotation marks
        unicodedata.lookup("LEFT DOUBLE QUOTATION MARK"): '"',
        unicodedata.lookup("RIGHT DOUBLE QUOTATION MARK"): '"',
        unicodedata.lookup("SINGLE LOW-9 QUOTATION MARK"): "'",
        unicodedata.lookup("DOUBLE LOW-9 QUOTATION MARK"): '"',
        unicodedata.lookup("LEFT SINGLE QUOTATION MARK"): "'",
        unicodedata.lookup("RIGHT SINGLE QUOTATION MARK"): "'",
        unicodedata.lookup("SINGLE LEFT-POINTING ANGLE QUOTATION MARK"): '"',
        unicodedata.lookup("SINGLE RIGHT-POINTING ANGLE QUOTATION MARK"): '"',
        unicodedata.lookup("LEFT-POINTING DOUBLE ANGLE QUOTATION MARK"): '"',
        unicodedata.lookup("rIGHT-POINTING DOUBLE ANGLE QUOTATION MARK"): '"',
        # special typographic marks
        unicodedata.lookup("END OF TEXT"): "",
        unicodedata.lookup("START OF HEADING"): "",
        unicodedata.lookup("HORIZONTAL ELLIPSIS"): "...",
        unicodedata.lookup("BULLET"): "*",
        unicodedata.lookup("EN DASH"): "-",
        unicodedata.lookup("EM DASH"): "-",
        unicodedata.lookup("SMALL TILDE"): "~",
        unicodedata.lookup("DAGGER"): "**",
        unicodedata.lookup("DOUBLE DAGGER"): "***",
        unicodedata.lookup("WHITE SMILING FACE"): "*",  # used as a bullet mark in some pdf documents
        # monetary units
        unicodedata.lookup("EURO SIGN"): "EUR",
        unicodedata.lookup("POUND SIGN"): "GBP",
        unicodedata.lookup("YEN SIGN"): "JPY",
        # fractions & math symbols
        unicodedata.lookup("VULGAR FRACTION ONE QUARTER"): "1/4",
        unicodedata.lookup("VULGAR FRACTION ONE HALF"): "1/2",
        unicodedata.lookup("VULGAR FRACTION THREE QUARTERS"): "*",  # used as a bullet mark in presentations
        unicodedata.lookup("VULGAR FRACTION ONE SEVENTH"): "1/7",
        unicodedata.lookup("VULGAR FRACTION ONE NINTH"): "1/9",
        unicodedata.lookup("VULGAR FRACTION ONE TENTH"): "1/10",
        unicodedata.lookup("VULGAR FRACTION ZERO THIRDS"): "0/3",
        unicodedata.lookup("VULGAR FRACTION ONE THIRD"): "1/3",
        unicodedata.lookup("VULGAR FRACTION TWO THIRDS"): "2/3",
        unicodedata.lookup("VULGAR FRACTION ONE FIFTH"): "1/5",
        unicodedata.lookup("VULGAR FRACTION TWO FIFTHS"): "2/5",
        unicodedata.lookup("VULGAR FRACTION THREE FIFTHS"): "3/5",
        unicodedata.lookup("VULGAR FRACTION FOUR FIFTHS"): "4/5",
        unicodedata.lookup("VULGAR FRACTION ONE SIXTH"): "1/6",
        unicodedata.lookup("VULGAR FRACTION FIVE SIXTHS"): "5/6",
        unicodedata.lookup("VULGAR FRACTION ONE EIGHTH"): "1/8",
        unicodedata.lookup("VULGAR FRACTION THREE EIGHTHS"): "3/8",
        unicodedata.lookup("VULGAR FRACTION FIVE EIGHTHS"): "5/8",
        unicodedata.lookup("FRACTION NUMERATOR ONE"): "1/",
        unicodedata.lookup("VULGAR FRACTION THREE EIGHTHS"): "3/8",
    },
)


def visible(char: str) -> bool:
    """Return true iff the given character is invisible."""
    return char in ("\n", "\t") or char.isprintable()


def normalize_text(text: str) -> str:
    """Normalize the text so as to lower case all, remove all special chars."""
    text: str = text.translate(REPLACEMENTS)
    text: str = unicodedata.normalize("NFD", text)
    text: str = "".join(c for c in text if not unicodedata.combining(c) and visible(c))
    text: str = unicodedata.normalize("NFC", text)
    return text.casefold()


def preprocess(text: str) -> str:
    """Preprocess the text to make it indexation ready."""
    text: str = normalize_text(text)
    text: str = "\n".join(line for line in text.splitlines() if not re.match(r"^\s*\d+(\.|\)|%)?\s*$", line))
    return text


if __name__ == "__main__":
    import pymupdf

    def get_text(fname: pathlib.Path, ocr: bool = False) -> str:
        """Extrait le texte d'un document."""
        if ocr:
            return "\n".join([page.get_text(textpage=page.get_textpage_ocr()) for page in pymupdf.open(fname)])
        return "\n".join([page.get_text() for page in pymupdf.open(fname)])

    text: str = get_text("./example/PV BF01 du 11.10.11.pdf")
    pathlib.Path("./dummy.txt").write_text(preprocess(text), encoding="utf8")
