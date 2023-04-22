"""
This module provides string formatting utilities for special cases where users have specific
formatting needs or preferences for their cloze deletions. It includes functions to parse
and format cloze text both before sending to LLMs and after receiving responses, handling
custom abbreviations, number formats, and special characters.

This is particularly useful when dealing with medical terminology, measurements, or other
domain-specific formatting requirements that might otherwise confuse LLMs or make the
content harder to read.
"""

import re
import sys
from utils.cloze_utils import iscloze

def cloze_input_parser(cloze: str) -> str:
    """edits the cloze from anki before sending it to the LLM. This is useful
    if you use weird formatting that mess with LLMs"""
    assert iscloze(cloze), f"Invalid cloze: {cloze}"

    cloze = cloze.replace("\xa0", " ")

    # make newlines consistent
    cloze = cloze.replace("<br/>", "<br>")
    cloze = cloze.replace("\r", "<br>")
    cloze = cloze.replace("<br>", "\n")

    # make spaces consitent
    cloze = cloze.replace("&nbsp;", " ")

    # author lists
    cloze = cloze.replace(" \n}}{{c", "}}\n{{c")

    # strip lines and remove empty lines
    cloze = "\n".join([c.strip() for c in cloze.splitlines() if c.strip()])

    # misc
    cloze = cloze.replace("&gt;", ">")
    cloze = cloze.replace("&ge;", ">=")
    cloze = cloze.replace("&lt;", "<")
    cloze = cloze.replace("&le;", "<=")

    assert iscloze(cloze), f"Invalid cloze: {cloze}"

    return cloze


def cloze_output_parser(cloze: str) -> str:
    """
    formats the cloze that were made easy to read by the LLM easy to
    read the by the author"""
    # strip
    cloze = cloze.strip()

    # useless punctuation
    cloze = cloze.replace("}},", "}}")
    cloze = cloze.replace("}}.", "}}")

    # make sure all newlines are consistent for now
    cloze = cloze.replace("</br>", "<br>")
    cloze = cloze.replace("\r", "<br>")
    cloze = cloze.replace("<br>", "\n")

    # make sure all spaces are consistent
    cloze = cloze.replace("&nbsp;", " ")

    # handle multiline cloze better
    cloze = cloze.replace("}} \n", "}}\n")
    cloze = cloze.replace("}}\n{{c", " \n}}{{c")
    cloze = cloze.replace("}} {{c", " }}{{c")

    # author quirks
    cloze = re.sub("sup[eé]rieure?s? à ", ">", cloze)
    cloze = re.sub("sup[eé]rieure?s? ", ">", cloze)
    cloze = re.sub("sup[eé]rieure?s? ou égale?s? ", ">=", cloze)
    cloze = re.sub("inf[eé]rieure?s? à ", "<", cloze)
    cloze = re.sub("inf[eé]rieure?s? ", "<", cloze)
    cloze = re.sub("inf[eé]rieure?s? ou égale?s? ", "<=", cloze)
    cloze = re.sub("intraveineuses?", "IV", cloze)
    cloze = cloze.replace("premières", "1eres")
    cloze = cloze.replace("première", "1ere")
    cloze = cloze.replace("premiers", "1er")
    cloze = cloze.replace("premier", "1er")
    cloze = cloze.replace("deuxièmes", "2emes")
    cloze = cloze.replace("deuxième", "2eme")
    cloze = cloze.replace("troisièmes", "3emes")
    cloze = cloze.replace("troisième", "3eme")
    cloze = cloze.replace("è", "e")
    # cloze = cloze.replace("é", "e")
    cloze = cloze.replace("\u0153", "oe")
    cloze = cloze.replace("ô", "o")
    cloze = cloze.replace("œ", "oe")
    cloze = cloze.replace("ë", "e")
    cloze = cloze.replace("ê", "e")
    cloze = cloze.replace("à", "a")
    cloze = cloze.replace("â", "a")
    cloze = cloze.replace("ç", "c")
    cloze = cloze.replace("ï", "i")
    cloze = cloze.replace("ù", "u")
    cloze = cloze.replace("}}.", "}}")
    cloze = cloze.replace(".}}", "}}")
    cloze = cloze.replace("millilitre ", "mL")
    cloze = cloze.replace("millilitres ", "mL")
    cloze = cloze.replace(" deux ", " 2 ")
    cloze = cloze.replace(" trois ", " 3 ")
    cloze = cloze.replace(" quatre ", " 4 ")
    cloze = cloze.replace(" cinq ", " 5 ")
    cloze = cloze.replace(" six ", " 6 ")
    cloze = cloze.replace(" sept ", " 7 ")
    cloze = cloze.replace(" huit ", " 8 ")
    cloze = cloze.replace(" dix ", " 10 ")
    cloze = cloze.replace(" onze ", " 11 ")
    cloze = cloze.replace(" douze ", " 12 ")
    cloze = cloze.replace(" treize ", " 13 ")
    cloze = cloze.replace(" quatorze ", " 14 ")
    cloze = cloze.replace(" quinze ", " 15 ")
    cloze = cloze.replace(" seize ", " 16 ")
    cloze = cloze.replace("contraindi", "contre-indi")

    # now use anki formatting for the newlines
    cloze = cloze.replace("\n", "<br>")

    return cloze

