"""Insert the 'Software and Reproducibility' subsection (V-D) and the 'Code and Data Availability' section
into an already-edited copy of the IEEE paper draft, without rebuilding it (so nobody's edits are lost).

    python paper/add_reproducibility_section.py path/to/IEEE_paper_draft.docx

Needs python-docx. Makes no change if the sections are already there.
"""

import sys

from docx import Document
from docx.enum.text import WD_COLOR_INDEX


def runs(p, parts):
    for text, flags in parts:
        r = p.add_run(text)
        if "Y" in flags:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        if "T" in flags:
            r.font.highlight_color = WD_COLOR_INDEX.TURQUOISE
    return p


def main():
    path = sys.argv[1]
    d = Document(path)
    paras = d.paragraphs
    if any("Software and Reproducibility" in p.text for p in paras):
        print("Already added; nothing to do.")
        return
    find = lambda text: next(p for p in paras if p.style.name == "Heading 1" and p.text.strip() == text)
    results_h, refs_h = find("VI. Results and Discussion"), find("References")

    results_h.insert_paragraph_before("D. Software and Reproducibility", style="Heading 2")
    runs(results_h.insert_paragraph_before("", style="Normal"), [
        ("Training and evaluation are written in Python 3.14 with scikit-learn 1.8.0, SciPy 1.17.0, NumPy 2.4.1, "
         "pandas 2.3.3 and Matplotlib 3.10.8, and sessions are logged over USB with pySerial 3.5. The firmware is "
         "C/C++ built with the Arduino-ESP32 core 4.0.0. The logistic regression is exported to C by our own script "
         "as a header of float32 constants; ", ""),
        ("[TEAM: name the converter each model used, e.g. micromlgen or emlearn, with its version]", "T"),
        (". Every model is trained and scored by the same evaluation script, which fixes the features, the "
         "session-level split and the metrics, so each number in Section VI can be regenerated from the recorded "
         "sessions. ", ""),
        ("[REVIEW: versions are from Aaron's machine; confirm everyone used the same ones]", "Y"),
    ])
    refs_h.insert_paragraph_before("Code and Data Availability", style="Heading 1")
    runs(refs_h.insert_paragraph_before("", style="Normal"), [
        ("The ESP32 firmware, the feature-extraction and evaluation code, and the 25 recorded sessions are "
         "available at ", ""),
        ("[link]", "Y"),
        (".", ""),
    ])
    d.save(path)
    print("Added V-D and the availability statement to", path)


if __name__ == "__main__":
    main()
