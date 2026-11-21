"""Build the group's IEEE-format paper draft (Word .docx).

    python build_paper.py            (needs: pip install python-docx)

Inputs (relative to Downloads):
  figures/paper/*.png             made by: set PAPER=1 and run make_figures.py
  paper/photos/setup_*.jpg        optional setup photos (top, side, blocked); placeholders if missing
Output:
  paper/IEEE_paper_draft.docx

Highlight key used in the draft:
  yellow     = drafted text or facts to review / verify before submission
  turquoise  = section owner still has to write this
"""

import os

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures", "paper")
PHOTOS = os.path.join(HERE, "photos")
OUT = os.path.join(HERE, "IEEE_paper_draft.docx")

FONT = "Times New Roman"
COL_W = 3.5          # IEEE column width (in)
FULL_W = 7.16        # full text width (in)

doc = Document()

# ---- page setup and styles -------------------------------------------------------------------
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin, sec.bottom_margin = Inches(0.75), Inches(1.0)
sec.left_margin = sec.right_margin = Inches(0.67)


def set_cols(section, num):
    sectPr = section._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sectPr.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), "360")          # 0.25 in between columns


def style_font(st, size, bold=False, italic=False, small_caps=False):
    st.font.name = FONT
    rf = st.element.rPr.rFonts
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(attr)) is not None:            # heading styles point at the theme font; drop it
            del rf.attrib[qn(attr)]
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(attr), FONT)
    st.font.size = Pt(size)
    st.font.bold, st.font.italic, st.font.small_caps = bold, italic, small_caps
    st.font.color.rgb = RGBColor(0, 0, 0)


normal = doc.styles["Normal"]
style_font(normal, 10)
pf = normal.paragraph_format
pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(0), 1.0
pf.first_line_indent = Inches(0.14)
pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

h1 = doc.styles["Heading 1"]
style_font(h1, 10, small_caps=True)
h1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
h1.paragraph_format.space_before, h1.paragraph_format.space_after = Pt(8), Pt(4)
h1.paragraph_format.first_line_indent = Inches(0)
h1.paragraph_format.keep_with_next = True

h2 = doc.styles["Heading 2"]
style_font(h2, 10, italic=True)
h2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
h2.paragraph_format.space_before, h2.paragraph_format.space_after = Pt(4), Pt(2)
h2.paragraph_format.first_line_indent = Inches(0)
h2.paragraph_format.keep_with_next = True

h3 = doc.styles["Heading 3"]
style_font(h3, 10, italic=True)
h3.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
h3.paragraph_format.space_before, h3.paragraph_format.space_after = Pt(2), Pt(0)
h3.paragraph_format.first_line_indent = Inches(0.14)
h3.paragraph_format.keep_with_next = True


# ---- helpers ---------------------------------------------------------------------------------
def add_runs(p, parts, size=None):
    """parts: strings or (text, flags). flags: b bold, i italic, Y yellow, T turquoise, c small caps,
    ^ superscript, _ subscript."""
    for part in parts:
        text, flags = (part, "") if isinstance(part, str) else part
        r = p.add_run(text)
        r.bold = "b" in flags or None
        r.italic = "i" in flags or None
        r.font.small_caps = "c" in flags or None
        if "^" in flags:
            r.font.superscript = True
        if "_" in flags:
            r.font.subscript = True
        if "Y" in flags:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        if "T" in flags:
            r.font.highlight_color = WD_COLOR_INDEX.TURQUOISE
        if size:
            r.font.size = Pt(size)
    return p


def P(*parts, align=None, indent=True, size=None, space_after=0, keep=False):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    if not indent:
        p.paragraph_format.first_line_indent = Inches(0)
    if space_after:
        p.paragraph_format.space_after = Pt(space_after)
    if keep:
        p.paragraph_format.keep_with_next = True
    return add_runs(p, parts, size)


def review(text):
    return (f"[REVIEW: {text}]", "Y")


def todo(owner, text):
    return (f"[{owner}: {text}]", "T")


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
counters = {"h1": 0, "h2": 0, "h3": 0, "fig": 0, "tab": 0}


def H1(title, numbered=True):
    if numbered:
        counters["h1"] += 1
        counters["h2"] = 0
        title = f"{ROMAN[counters['h1'] - 1]}. {title}"
    doc.add_paragraph(title, style="Heading 1")


def H2(title):
    counters["h2"] += 1
    counters["h3"] = 0
    doc.add_paragraph(f"{chr(64 + counters['h2'])}. {title}", style="Heading 2")


def H3(title, *extra):
    counters["h3"] += 1
    p = doc.add_paragraph(style="Heading 3")
    add_runs(p, [f"{counters['h3']}) {title}: "] + list(extra))
    return p


def bullets(items, size=None):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.first_line_indent = Inches(0)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        add_runs(p, it if isinstance(it, list) else [it], size)
        for r in p.runs:
            r.font.name = FONT


def new_section(cols):
    s = doc.add_section(WD_SECTION.CONTINUOUS)
    set_cols(s, cols)
    return s


def caption_fig(text_parts):
    counters["fig"] += 1
    p = P((f"Fig. {counters['fig']}. ", ""), *text_parts, indent=False, size=8, space_after=6,
          align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    return counters["fig"]


def figure(path, width, caption_parts, full=False):
    """Insert a figure (full=True spans both columns via a one-column section)."""
    if full:
        new_section(1)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.keep_with_next = True
    if os.path.exists(path):
        p.add_run().add_picture(path, width=Inches(width))
    else:
        add_runs(p, [(f"[FIGURE MISSING: {os.path.basename(path)}]", "Y")])
    n = caption_fig(caption_parts)
    if full:
        new_section(2)
    return n


def table_caption(title):
    counters["tab"] += 1
    P((f"TABLE {ROMAN[counters['tab'] - 1]}", "c"), align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, size=8, keep=True)
    P((title, "c"), align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, size=8, space_after=2, keep=True)
    return counters["tab"]


def table(rows, widths, header=True, size=8):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.width = Inches(widths[j])
            cell.paragraphs[0].paragraph_format.first_line_indent = Inches(0)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT
            parts = val if isinstance(val, list) else [val]
            if header and i == 0:
                parts = [(x, "b") if isinstance(x, str) else x for x in parts]
            add_runs(cell.paragraphs[0], parts, size)
            if i < len(rows) - 1:                   # keep the whole table (and its caption) together
                cell.paragraphs[0].paragraph_format.keep_with_next = True
        trPr = t.rows[i]._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def photo_figure():
    """Three setup photos side by side across the full page width (one-column section)."""
    names = [("setup_top", "(a) top view: ESP32 on the breadboard, fan bridged across two textbooks with tape at the "
                           "corners, MPU6050 (red LED) taped to the left edge of the frame;"),
             ("setup_side", "(b) side view of the tape fixtures and wiring;"),
             ("setup_blocked", "(c) blocked-airflow condition: an index card covering about half of the intake.")]
    found = []
    for stem, desc in names:
        for ext in (".jpg", ".jpeg", ".png"):
            path = os.path.join(PHOTOS, stem + ext)
            if os.path.exists(path):
                found.append((stem, path, desc))
                break
        else:
            found.append((stem, None, desc))
    # Phone photos are often stored sideways with an EXIF rotation tag. Make upright, smaller copies so the
    # size calculation and Word agree (needs Pillow).
    from PIL import Image, ImageOps
    fixed_dir = os.path.join(PHOTOS, "_upright")
    os.makedirs(fixed_dir, exist_ok=True)
    ratios = []
    for k, (stem, path, desc) in enumerate(found):
        if path:
            im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            im.thumbnail((1400, 1400))
            out = os.path.join(fixed_dir, stem + ".jpg")
            im.save(out, quality=88)
            found[k] = (stem, out, desc)
            ratios.append(im.width / im.height)
        else:
            ratios.append(0.75)
    gap = 0.08
    height = (FULL_W - 0.1 - 2 * gap) / sum(ratios)
    widths = [r * height for r in ratios]

    new_section(1)
    # photos inline in one centered paragraph (a table would force equal column widths and crop them)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.keep_with_next = True
    for j, (stem, path, desc) in enumerate(found):
        if j:
            p.add_run(" ")
        if path:
            p.add_run().add_picture(path, height=Inches(height))
        else:
            add_runs(p, [(f"[PHOTO PLACEHOLDER: save as paper/photos/{stem}.jpg and re-run build_paper.py]", "Y")], 8)
    # (a) (b) (c) centered under each photo with center tab stops
    labels = doc.add_paragraph()
    labels.paragraph_format.first_line_indent = Inches(0)
    labels.paragraph_format.keep_with_next = True
    x = 0.0
    for j, w in enumerate(widths):
        labels.paragraph_format.tab_stops.add_tab_stop(Inches(x + w / 2), alignment=1)   # 1 = center
        x += w + gap
    add_runs(labels, ["\t(a)\t(b)\t(c)"], 8)
    n = caption_fig(["Test setup. " + " ".join(d for _, _, d in found)])
    new_section(2)
    return n


# ---- title block (one column) -----------------------------------------------------------------
set_cols(sec, 1)
t = P(("Which Model Catches the Fault? Comparing Machine Learning Models for TinyML Anomaly "
       "Detection in Predictive Maintenance of Power Equipment", ""),
      align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, size=24, space_after=10)

authors = [("Aaron Arnold", "Logistic regression"), ("Iritza Khan", "Linear SVM"),
           ("Angelina Guasti", "Isolation forest"), ("Devon Biscardi", "Decision tree")]
tbl = doc.add_table(rows=1, cols=4)
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
for j, (name, _) in enumerate(authors):
    c = tbl.cell(0, j)
    c.width = Inches(1.79)
    par = c.paragraphs[0]
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.first_line_indent = Inches(0)
    add_runs(par, [name], 11)
    for line in ("[Department]", "[University]", "[City, State, USA]", "[email]"):
        q = c.add_paragraph()
        q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        q.paragraph_format.first_line_indent = Inches(0)
        add_runs(q, [(line, "iT")], 9)

P(("Draft key (delete before submission): ", "bY"),
  ("yellow = drafted text or facts to review/verify; ", "Y"),
  ("turquoise = section owner still has to write this. ", "T"),
  ("Each model owner writes their model's subsections in Sections IV-C, VI and the results table.", "Y"),
  align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, size=8, space_after=6)

new_section(2)

# ---- abstract --------------------------------------------------------------------------------
P(("Abstract", "bi"), ("—Power outages cost the U.S. economy an estimated $150 billion per year, and equipment "
  "failures cause about 25% of them ", "b"), review("add source"),
  (". Low-cost Internet of Things (IoT) sensors that analyze vibration on the device itself can flag failing "
   "equipment early while sending only small alerts over the network. This paper compares four machine learning "
   "models, logistic regression, a linear support vector machine (SVM), an isolation forest and a decision tree, "
   "for detecting faults in a 12 V cooling fan, a stand-in for transformer cooling fans and pumps, from MPU6050 "
   "accelerometer data processed on an ESP32 microcontroller. All models share one feature set (RMS, peak, "
   "standard deviation, kurtosis, crest factor and four FFT band energies over 1-s windows) and one session-level "
   "train/test split. We report precision, recall, F1 score, false alarm rate and ROC-AUC on three labeled faults "
   "(imbalance, loose mount and blocked airflow), recall on a fault never seen in training (blade rub), and "
   "on-device flash, RAM and inference time. ", "b"),
  todo("TEAM", "one sentence with the headline comparison once all four models are in"),
  (" The logistic regression detects every labeled fault in the held-out test sessions (F1 = 1.00), flags all "
   "unseen blade-rub windows, and needs 112 bytes of parameters and 10–18 µs per inference on the ESP32; live "
   "tests show it is sensitive to changes in how the fan is mounted.", "b"),
  indent=False, size=9, space_after=4)
P(("Index Terms", "bi"), ("—TinyML, anomaly detection, predictive maintenance, vibration analysis, ESP32, "
                          "edge computing, Internet of Things.", "b"), indent=False, size=9, space_after=4)

# ---- I. Introduction -------------------------------------------------------------------------
H1("Introduction")
P("Power outages cost the U.S. economy about $150 billion per year, and equipment failures cause about a quarter "
  "of them ", review("cite a source for both numbers, e.g., DOE/EIA or an industry report"),
  ". Much of this equipment, such as transformer cooling fans and pumps, degrades gradually: bearings wear, "
  "blades become unbalanced, mounts loosen and airflow is obstructed. Each of these changes the way the machine "
  "vibrates, so a vibration sensor can reveal a developing fault well before it causes an outage ",
  review("cite, e.g., [5]"), ".")
P("IoT sensors give operators real-time insight into equipment health, and predictive maintenance turns that "
  "insight into repairs scheduled before a failure [3]. Running the detection model on the sensor itself, an "
  "approach known as TinyML, adds three practical benefits [1]. First, connectivity: only small alerts leave the "
  "device, so it works over weak links at remote sites. Second, security: raw data stays on the device. Third, "
  "integration: a cheap sensor can be retrofitted onto old equipment that has no monitoring at all.")
P("The open question is which model to run. Many candidate models exist, from simple linear classifiers to tree "
  "ensembles and unsupervised anomaly detectors, and it is not clear which one gives the best detection on "
  "hardware as small as a microcontroller [2]. This paper asks: ",
  ("which machine learning models detect equipment faults most accurately when they run on a low-cost IoT "
   "microcontroller (ESP32), and does the extra memory and computation of more complex models buy better "
   "detection?", "i"), " We break this into three sub-questions:")
bullets(["Which models give the best precision, recall and F1 score on known faults?",
         "Which models still catch a fault type they never saw during training?",
         "Which models fit on the ESP32 at all, and how fast do they make a prediction?"])
P("Our contributions are: ", review("keep only what is finished by submission"))
bullets(["a low-cost, reproducible test bed (ESP32, MPU6050 and a 12 V fan) with three labeled fault conditions "
         "and one held-out fault, recorded as 25 one-minute sessions at 1 kHz;",
         "a shared feature-extraction and evaluation pipeline that applies the same features and the same "
         "session-level data split to every model;",
         "a comparison of four models on detection accuracy, generalization to an unseen fault and on-device cost;",
         "live on-device tests, including faults introduced while the detector was running."])
P("Section II reviews related work, Section III describes the hardware and data pipeline, Section IV the data, "
  "features and models, Section V the experimental setup, and Section VI the results. Section VII discusses "
  "limitations and Section VIII concludes.")

# ---- II. Related work ------------------------------------------------------------------------
H1("Related Work")
P(review("drafted from the reference titles in our outline; everyone please check each sentence against the paper "
         "and add the 8–12 extra IEEE Xplore references the outline asks for"))
H2("TinyML Anomaly Detection")
P("Antonini et al. [1] present an adaptable, unsupervised TinyML anomaly detection system for extreme industrial "
  "environments. Almaini et al. [6] apply TinyML to acoustic anomaly detection in IoT sensor networks, and "
  "Martinez-Rau et al. [7] address TinyML anomaly detection for industrial machines with periodic duty cycles. "
  "Wang et al. [10] study TinyML-based out-of-distribution anomaly detection with time synchronization in "
  "industrial IoT. Liu et al. [4] introduced the isolation forest, which isolates anomalies with random splits "
  "and needs only normal data for training.")
H2("Machine Learning for Predictive Maintenance")
P("Mohd Ghazali and Rahiman [5] review vibration analysis for machine monitoring and diagnosis. Kolok et al. [3] "
  "build low-cost IoT predictive maintenance on vibration data, Njor et al. [9] review the TinyML stack for "
  "predictive maintenance, and Trilles et al. [2] map the literature on anomaly detection with the Artificial "
  "Intelligence of Things.")
H2("Model Comparisons on Microcontrollers")
P("Katib et al. [8] combine deep learning with TinyML for real-time anomaly detection on IoT consumer devices, "
  "and Sharma et al. [11] optimize a stacking-based TinyML model for attack detection in IoT networks. ",
  todo("TEAM", "add direct model-comparison papers and say how our study differs (same features and split, "
               "on-device measurements, an unseen fault)"))

# ---- III. System design ----------------------------------------------------------------------
H1("System Design")
H2("Hardware")
P("The test bed uses an ESP32 DevKit (ESP-WROOM-32, 240 MHz dual-core, 320 KB RAM, 4 MB flash) and an "
  "MPU6050 three-axis MEMS accelerometer connected over I²C at 400 kHz. A 12 V, 120 mm, 4-pin PC fan "
  "powered from a bench 12 V supply stands in for a transformer cooling fan. The ESP32 sets the fan speed with "
  "a 25 kHz PWM signal (duty 204/255, 80%) and counts the fan's tachometer pulses (2 per revolution) through a "
  "10 kΩ pull-up to 3.3 V, so every recording also logs fan speed. The fan, supply and ESP32 share a common "
  "ground. ", review("confirm board variant, fan model (Arctic, 7 blades) and supply"))
photo_fig = photo_figure()
H2("Sensor Setup")
P(f"The fan rests face-down across two hardcover textbooks so that the intake and exhaust stay open, and painter's "
  f"tape holds each of its four corners to the books (Fig. {photo_fig}). The MPU6050 is taped flat to one corner "
  "of the fan frame in the same position and orientation for every session. The sensor runs at its ±8 g range "
  "(4096 counts/g) with the digital low-pass filter at about 184 Hz and an output rate of 1 kHz. The ±8 g range "
  "replaced an initial ±2 g setting after the imbalance fault clipped the sensor. At the chosen duty the fan "
  "turns at about 1,690 RPM, a spin frequency of about 28 Hz.")
P("Logging fan speed proved essential. In an earlier data set the normal sessions ran at about 950 RPM and every "
  "fault session at 690–800 RPM because of a faulty ground; a classifier given only the fan speed separated "
  "normal from fault perfectly. After the ground was fixed, every condition runs within about 1% of the same "
  "speed, and that data set is the one used in this paper.")
H2("Data Pipeline")
P("A logging sketch reads the accelerometer every 1 ms and streams raw counts over USB serial at 921,600 baud. "
  "A Python script (record_session.py) records one labeled 60-s session per run, saving a CSV of timestamps, "
  "the three axes and the latest fan speed, plus a JSON file with the sample rate, timing gaps, fan speed, "
  "sensor range, fan duty, number of clipped readings and notes. Sessions average 999.7 Hz with at most 14 "
  "timing gaps longer than 1.5 ms per minute and no clipped readings.")

# ---- IV. Methodology -------------------------------------------------------------------------
H1("Methodology")
H2("Data Collection")
P("Five conditions were recorded, five 60-s sessions each, all on November 19, 2026 (Table I). The held-out "
  "fault is never used for training; it only tests whether a model notices something abnormal that it was "
  "not taught.")
table_caption("Recorded conditions")
table([["Condition", "How it was created", "RPM"],
       ["Normal", "Fan running as is, all corners taped", "1,688"],
       ["Imbalance", "Twist tie (~4 cm) on one blade, ~1 cm from the tip", "1,694"],
       ["Loose mount", "Bottom-left corner untaped", "1,687"],
       ["Blocked airflow", "Index card over half of the intake", "1,686"],
       ["Held-out: blade rub", "~1 mm of index card touching the blade tips", "1,687"]],
      [0.95, 2.05, 0.5])
P("Each condition has a distinct signature in the frequency domain (Fig. 2). Imbalance raises the 1× spin "
  "component at 28 Hz about 30-fold, blocked airflow adds a tone at the blade-pass frequency (7 × 28 ≈ 195 Hz), "
  "and the loose mount and the blade rub add broadband vibration above 200 Hz. The rub changed as the card wore: "
  "its first session shows the blade-pass tone while later sessions mostly add high-frequency buzz. ",
  review("outline asked for sessions on different days; all were recorded on one day, so list it as a limitation"))
spec_fig = figure(os.path.join(FIG, "1_spectra.png"), FULL_W,
                  ["Mean FFT amplitude spectrum of the acceleration magnitude for one session of each condition "
                   "(blue) against a normal session (grey). Dashed line: 1× spin frequency; dotted line: blade-pass "
                   "frequency (7 blades)."], full=True)
H2("Feature Extraction")
P("All four models use the same features, computed by the shared script (features.py). Each session is cut into "
  "non-overlapping 1-s windows (1000 samples) of the acceleration magnitude in m/s². Each window yields nine "
  "features: RMS, peak, standard deviation, excess kurtosis, crest factor (peak/RMS) and the mean FFT energy in "
  "four bands, 5–25, 25–75, 75–200 and 200–500 Hz, computed after removing the window mean. This gives 59 or "
  "60 windows per session and 1,476 windows in total. ",
  review("features.py must use SCALE_MS2 = 9.80665/4096 for the ±8 g data; it still has 16384. Also consider "
         "removing the mean before rms and peak, which currently include gravity"))
H2("Models")
H3("Logistic regression", ("Aaron Arnold", "i"))
P("Logistic regression estimates the probability that a window is a fault as")
P(("p = 1 / (1 + e", "i"), ("−s", "i^"), ("),   s = b + Σ", "i"), ("i", "i_"), (" w", "i"), ("i", "i_"),
  (" (x", "i"), ("i", "i_"), (" − μ", "i"), ("i", "i_"), (") / σ", "i"), ("i", "i_"), ("\t(1)", ""),
  align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
P("where x is the nine-feature vector, μ and σ are the feature means and standard deviations of the "
  "training set, w the weights and b the intercept; a window is a fault when p ≥ 0.5, i.e. s ≥ 0. We train it "
  "with scikit-learn (L-BFGS, L2 penalty) on standardized features with balanced class weights, because fault "
  "windows outnumber normal ones three to one. The regularization strength C is chosen from 10⁻³ to 10³ by "
  "four-fold cross-validation grouped by session on the training sessions only, maximizing F1. The binary "
  "model selected C = 10. A second, multinomial model with the same features (C = 0.1) predicts which of the "
  "four training conditions a window belongs to.", indent=False)
P("Deployment needs no converter: the model is one dot product. The means, standard deviations, weights and "
  "intercept are exported as 28 float32 constants (112 bytes) in a C header, and (1) is evaluated directly on "
  "the ESP32.")
H3("Linear SVM", ("Iritza Khan", "i"))
P(todo("Iritza", "model description, hyperparameters and tuning, conversion to C (micromlgen/emlearn), model size"))
H3("Isolation forest", ("Angelina Guasti", "i"))
P(todo("Angelina", "model description (trained on normal windows only), number of trees, contamination/threshold "
                   "choice, how the trees are exported to C"))
H3("Decision tree", ("Devon Biscardi", "i"))
P(todo("Devon", "model description, depth and pruning, tuning, conversion to C, model size"))

# ---- V. Experimental setup -------------------------------------------------------------------
H1("Experimental Setup")
H2("Data Splits")
P("All models use the same split. Sessions 1–4 of each training condition form the training set (945 windows) "
  "and the last session of each condition forms the test set (236 windows), so no test session is ever seen in "
  "training. The five held-out sessions (295 windows) are used only for testing. Because one test session per "
  "condition is a small sample, we also repeat the evaluation over five rounds in which round k tests on "
  "session k of every condition, and report the mean and standard deviation. Hyperparameters are tuned inside "
  "the training sessions only.")
H2("Metrics")
P("The main task is normal versus fault, since the isolation forest can only output normal or anomaly. We report "
  "precision (of the alerts sent, how many were real faults), recall (of the real faults, how many were caught), "
  "F1 score, false alarm rate (the fraction of normal windows flagged), ROC-AUC and binary confusion matrices. "
  "The supervised models are also scored on which fault occurred (macro F1 and a multiclass confusion matrix). "
  "Generalization is measured as recall on the held-out fault.")
H2("On-Device Deployment and Tests")
P("Each model is deployed in one ESP32 sketch that samples 1-s windows at 1 kHz, computes the nine features in "
  "C and runs the model. Because 1000 samples is not a power of two, the band energies come from a direct DFT "
  "of bins 5–499 using a cosine/sine lookup table. A verification mode receives recorded windows from the PC "
  "and returns the board's features and prediction, which a script compares with Python. A live mode classifies "
  "the running fan and raises an alert after three fault windows in a row. We record flash and RAM use, "
  "feature-extraction time and inference time. ",
  todo("TEAM", "confirm every model uses this same sketch and test procedure"))

# ---- VI. Results and discussion --------------------------------------------------------------
H1("Results and Discussion")
H2("Detection Results")
P("Table II summarizes all four models on the shared test split. ",
  todo("TEAM", "fill in your row; then write the comparison and the F1 bar chart and F1-vs-size scatter from the "
               "outline"))
new_section(1)
table_caption("Detection and on-device results on the shared test split")
table([["Model", "Precision", "Recall", "F1", "False alarm", "ROC-AUC", "Held-out recall", "Parameters",
        "Inference"],
       ["Logistic regression", "1.000", "1.000", "1.000", "0.000", "1.000", "1.000", "112 B", "10–18 µs"],
       ["Linear SVM"] + [("TBD", "T")] * 8,
       ["Isolation forest"] + [("TBD", "T")] * 8,
       ["Decision tree"] + [("TBD", "T")] * 8],
      [1.25, 0.66, 0.6, 0.55, 0.75, 0.7, 0.95, 0.8, 0.8])
new_section(2)
P(("Logistic regression. ", "i"),
  "On the shared split the logistic regression classifies all 236 test windows correctly (Fig. 3, left): "
  "precision, recall and F1 are 1.000 with no false alarms. Over the five session-level rounds it averages "
  "F1 = 0.993 ± 0.016, a false alarm rate of 0.3% ± 0.7% and ROC-AUC = 0.997 ± 0.007. The multinomial model identifies the fault type "
  "with macro F1 = 0.983; its only errors are four blocked-airflow windows labeled as loose mount (Fig. 3, "
  "centre). As a check on the earlier fan-speed problem, the same classifier given only the fan speed reaches "
  "just F1 = 0.67 on the new data.")
P("These near-perfect offline scores come from one fan, one setup and one afternoon of recordings, and the "
  "weights show what the model relies on (Fig. 4). The largest weight is on the 200–500 Hz band, followed by "
  "kurtosis and the 75–200 Hz band, while the 25–75 Hz band that contains the imbalance's 28 Hz peak receives "
  "almost none. In training, imbalance also raised the high-frequency energy, and the model learned that side "
  "effect instead of the main signature. ", review("a log transform of the band energies before standardizing "
                                                  "would likely fix this; we kept the deployed model unchanged"))
todo_par = P(todo("Iritza / Angelina / Devon", "your model's detection results and confusion matrix here"))
cm_fig = figure(os.path.join(FIG, "2_confusion_matrices.png"), FULL_W,
                ["Logistic regression on the shared test split. Left: normal vs. fault. Centre: which fault. "
                 "Right: how the multinomial model labels the held-out blade rub, which it was never trained on."],
                full=True)
w_fig = figure(os.path.join(FIG, "3_feature_weights.png"), COL_W,
               ["Weights of the deployed logistic regression on the standardized features (blue pushes towards "
                "fault, orange towards normal)."])
H2("Generalization to an Unseen Fault")
P("The logistic regression flags all 295 held-out blade-rub windows as faults (recall = 1.000 on the shared "
  "split and in all five rounds). The multinomial model labels 83% of them as loose mount and 17% as blocked "
  "airflow (Fig. 3, right), which matches their spectra: the rub adds high-frequency buzz like the loose "
  "mount, and in its first session a blade-pass tone like the blocked intake. The rub therefore resembles faults "
  "the model was trained on, so this test is easier than a truly novel fault, and high held-out recall should be "
  "read with that in mind. ", todo("TEAM", "compare with the isolation forest, the key model for this test"))
H2("On-Device Feasibility")
P("On the ESP32 the logistic regression's C implementation matched Python on all 531 recorded windows tested "
  "(100% agreement, largest probability difference 6 × 10⁻⁵). Table III lists its cost. Inference takes "
  "10–18 µs; nearly all of the per-window time is feature extraction (42.8 ms), which every model shares.")
table_caption("Logistic regression on the ESP32")
table([["Measure", "Value"],
       ["Model parameters", "112 B (28 float32)"],
       ["Inference time per window", "10–18 µs"],
       ["Feature extraction per 1-s window", "42.8 ms"],
       ["Flash vs. logging sketch", "+8.7 KB"],
       ["RAM vs. logging sketch", "+22 KB (16 KB feature buffers)"],
       ["ESP32 = Python predictions", "531 / 531 windows"]],
      [2.0, 1.5])
P(todo("Iritza / Angelina / Devon", "your model's flash, RAM and inference time; add rows or a column"))
H2("Live Tests on the ESP32")
P("We then ran the logistic regression on the live fan (Fig. 5). It classified a normal run correctly in all 29 "
  "windows, imbalance in 28 of 28 and a rattling loose mount in 28 of 29. With a weaker imbalance (a looser "
  "twist tie, about 60% of the recorded vibration) only 86% of windows were flagged, because the model relies on "
  "a side effect of imbalance rather than its main signature. A loose corner that sat still produced no extra "
  "vibration and was never flagged; no vibration-based model can detect a fault that does not change the "
  "vibration.")
live_fig = figure(os.path.join(FIG, "5_live_tests.png"), COL_W,
                  ["Share of 1-s windows classified correctly in 30-s live tests on the ESP32. In the last run the "
                   "untaped corner sat still and the vibration matched normal."])
P("When a fault was introduced while the detector was running (Fig. 6), it raised no false alarms beforehand and "
  "alerted 3.1 s after the fault appeared in every case, the minimum allowed by the three-window rule. "
  "Removing the tape from the loose corner caused a single fault window but no alert; the corner then took about "
  "7 s to start rattling. In the blocked-airflow run the card fluttered or lightly touched the blades, so the "
  "vibration was stronger than in the recordings. ",
  review("re-run blocked airflow with the card fully flat if time allows"))
tl_fig = figure(os.path.join(FIG, "6_live_timelines.png"), FULL_W,
                ["Fault probability over time on the ESP32 when a fault was added during the run. Dotted lines mark "
                 "the change and red lines the alert (three fault windows in a row). In the blocked-airflow run the "
                 "card fluttered or touched the blades."], full=True)
P("The live tests also exposed sensitivity to mounting. Before one normal test the fan had been re-seated "
  "after the fault experiments; the detector flagged every window, and its features matched the recorded "
  "loose-mount condition. After all four corners were re-taped, the same test passed 29/29. A model trained on "
  "one setup can mistake a small mounting change for a fault, so normal data should be recorded across several "
  "re-mountings, as the outline intended.")

# ---- VII. Limitations ------------------------------------------------------------------------
H1("Limitations and Future Work")
bullets([["A single small fan in a lab, and lab-made faults; real equipment is larger and has more vibration "
          "sources."],
         ["All sessions were recorded on one day with one mounting, and each condition has only one test session, "
          "so offline scores are optimistic."],
         ["Detection depends on how the fan is mounted (Section VI-D)."],
         ["The MPU6050 filter (about 184 Hz) weakens the 200–500 Hz band that the logistic regression relies on."],
         ["Power consumption was not measured."],
         ["The shared rms and peak features include gravity, so they partly measure sensor tilt rather than "
          "vibration."],
         ["Future work: log-scaled band energies, narrow bands at the spin and blade-pass frequencies, data from "
          "several days and setups, larger equipment, and more models such as small neural networks."],
         [todo("TEAM", "add limitations from your models")]])

# ---- VIII. Conclusion ------------------------------------------------------------------------
H1("Conclusion")
P(todo("TEAM", "answer the research question across all four models"), " ",
  ("For the logistic regression: a 112-byte linear model on nine shared features detects lab-made fan faults "
   "on an ESP32 in microseconds, with feature extraction dominating the cost, and alerts about 3 s after a fault "
   "appears. Its weakness is not size or speed but what it learned: it relies on high-frequency side effects and "
   "is sensitive to mounting changes.", "Y"))

# ---- References ------------------------------------------------------------------------------
H1("References", numbered=False)
refs = [
    'M. Antonini, M. Pincheira, M. Vecchio, and F. Antonelli, "An adaptable and unsupervised TinyML anomaly '
    'detection system for extreme industrial environments," Sensors, 2023.',
    'S. Trilles, S. S. Hammad, and D. Iskandaryan, "Anomaly detection based on Artificial Intelligence of Things: '
    'A systematic literature mapping," Internet of Things, vol. 25, 2024.',
    'P. Kolok, M. Hodoň, P. Ševčík, L. Hotz, and N. Remy, "Low-cost IoT-based predictive maintenance using '
    'vibration," Sensors, vol. 25, no. 21, Art. no. 6610, Oct. 2025, doi: 10.3390/s25216610.',
    'F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in Proc. 8th IEEE Int. Conf. Data Mining (ICDM), '
    'Pisa, Italy, 2008, pp. 413–422, doi: 10.1109/ICDM.2008.17.',
    'M. H. Mohd Ghazali and W. Rahiman, "Vibration analysis for machine monitoring and diagnosis: A systematic '
    'review," Shock and Vibration, vol. 2021, Art. no. 9469318, 2021, doi: 10.1155/2021/9469318.',
    'A. Almaini, J. Folz, and G. Ashour, "TinyML for acoustic anomaly detection in IoT sensor networks," in Proc. '
    '5th Int. Conf. Electrical, Computer, Communications and Mechatronics Engineering (ICECCME), Oct. 2025, '
    'pp. 1–5, doi: 10.48550/arXiv.2603.26135.',
    'L. S. Martinez-Rau, Y. Zhang, B. Oelmann, and S. Bader, "TinyML anomaly detection for industrial machines '
    'with periodic duty cycles," in Proc. IEEE Sensors Applications Symp. (SAS), Jul. 2024, pp. 1–6, '
    'doi: 10.1109/SAS60918.2024.10636584.',
    'I. Katib, E. Albassam, S. A. Sharaf, and M. Ragab, "Safeguarding IoT consumer devices: Deep learning with '
    'TinyML driven real-time anomaly detection for predictive maintenance," Ain Shams Eng. J., vol. 16, no. 2, '
    'Art. no. 103281, Feb. 2025, doi: 10.1016/j.asej.2025.103281.',
    'E. Njor, M. A. Hasanpour, J. Madsen, and X. Fafoutis, "A holistic review of the TinyML stack for predictive '
    'maintenance," IEEE Access, 2024, doi: 10.1109/ACCESS.2024.3512860.',
    'M. Wang, W. Bian, Q. Xie, M. Lv, and Q. Du, "TinyML-based OOD anomaly detection and time synchronization '
    'optimization: Enhancing the spatiotemporal consistency of anomalous data in industrial IoT," Sensor Review, '
    'vol. 46, no. 4, pp. 569–582, Dec. 2025, doi: 10.1108/SR-07-2025-0503.',
    'A. Sharma, S. Rani, and M. Shabaz, "An optimized stacking-based TinyML model for attack detection in IoT '
    'networks," PLoS ONE, vol. 20, no. 8, Art. no. e0329227, 2025, doi: 10.1371/journal.pone.0329227.',
]
for i, ref in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(0.3))
    add_runs(p, [f"[{i}]\t{ref}"], 8)
P(review("check every reference against the original (authors, venue, volume, pages, DOI); the Almaini DOI "
         "looks like an arXiv ID; add the extra references and renumber in order of first citation"),
  indent=False, size=8)

doc.save(OUT)
print("wrote", OUT)
