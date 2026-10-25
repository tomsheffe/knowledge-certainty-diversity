"""Annotated conversation excerpts from Experiments 3 and 4, drawn as chat windows.

Four cases, each identified by its source row so the selection is traceable:

  A  Exp 4, row 145   Student defers to a confident LLM and is right to.
  B  Exp 4, row  34   LLM defers to the student and is right to.
  C  Exp 4, row 378   Student defers to a confident LLM and should not have.
  D  Exp 3, group E3-G20i, Q38   A confident dissenter converts the majority.

A and C are a matched pair: in both, the student raises a correct clinical
objection and the LLM concedes the sub-point while holding its answer. The
dialogues are structurally the same; only the ground truth differs. Nothing
on the surface of either conversation distinguishes the case where deference
helps from the case where it hurts — which is the argument for estimating
synergy before a conversation rather than after it.

D is the human-trio case. It is
the configuration the framework predicts should pay: the dissenter is more
confident (5) than either member of the majority (3 and 2) and is the one
who is right. Confidence is never stated as a number anywhere in the
exchange — the least confident participant simply says "I honestly have no
idea", which is what a recorded rating of 2 sounds like in words — so the
panel shows confident-knowledge being exchanged in ordinary speech rather
than recited to a protocol.

Layout. Each panel is a chat
window: a header bar naming the conversation, a pinned question card with
the correct answer, a "discussion" divider, then message bubbles with tails
and avatars — robot icon on the left for the LLM, person icon on the right
for the student; in the trio, three numbered person icons in a group-chat
layout — and finally the outcome as a centred system message, one line per
agent. Speaker names inside bubbles are gone; the avatar and the side do
that work. Icons are drawn from matplotlib primitives so the figure has no
external assets.

Sources. Exp 4 numerical outcomes: ../data/medcom2_bot_post_conversation_updated.csv.
Exp 3 answers and confidence: ../data/medcom1_fulldata.csv. Released Hebrew
transcripts: ../data/conversations/experiment3.csv and experiment4.csv.
The translated excerpts are embedded below.

Speaker mapping for D. The transcript labels speakers i1/i2/i3; the answer
data identifies subjects E3-P056, E3-P058, E3-P060. The mapping is confirmed by
content, not assumed: i3 opens with "A" and gives the NSAID rule, and E3-P060
is the lone participant who answered A; i2 says "I honestly have no idea"
and E3-P058 holds the lowest recorded confidence in the trio. Participants are
numbered, never named.

Translation and de-identification. Conversations were conducted in Hebrew;
the English below is a human translation held next to the Hebrew source so
it can be checked. These excerpts contain no names or other identifiers
Long LLM turns
are abridged at the marked ellipses and C omits a closing courtesy line;
abridgement removes elaboration, never a claim or a conclusion. Raw message
counts: rows 145 and 34 have 3 messages each, row 378 has 7 (6 shown), and
E3-G20i/Q38 has 6 — no turns are merged.

Every number in the outcome lines is read from the CSVs at draw time, so the
figure cannot drift from the data.

Writes ../outputs/figures/figure_conversations.png and its caption.
"""
import math
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Wedge

HERE = os.path.dirname(os.path.abspath(__file__))
DATA4 = os.path.join(HERE, '..', 'data', 'medcom2_bot_post_conversation_updated.csv')
DATA3 = os.path.join(HERE, '..', 'data', 'medcom1_fulldata.csv')
OUT_DIR = os.path.join(HERE, '..', 'outputs', 'figures')

plt.rcParams['font.sans-serif'] = ['Avenir Next', 'Avenir', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'

# ---------------------------------------------------------------- palette
C_INK, C_MUTED, C_FAINT = '#1F2328', '#6B7280', '#9CA3AF'
C_OK, C_BAD = '#1B7A3D', '#C0392B'
C_OK_BG, C_BAD_BG = '#DCEFE2', '#F9E0DC'
C_WALL = '#F3F4F6'          # chat wallpaper
C_HEADER = '#2F3640'        # header bar
C_QCARD = '#FFFFFF'         # pinned question card
C_SYS = '#E5E7EB'           # system-message pill

# speaker -> (name, kind, bubble fill, accent, align right)
SPEAKERS = {
    'llm':   ('LLM',           'robot',  '#E4ECF8', '#4C7DBF', False),
    'human': ('Student',       'person', '#DCF3E4', '#3C9A5F', True),
    'p1':    ('Participant 1', 'person', '#EFEAE0', '#A8956B', False),
    'p2':    ('Participant 2', 'person', '#EBE3F5', '#8E6BB8', False),
    'p3':    ('Participant 3', 'person', '#DCEFEA', '#3E9C88', False),
}

EXP_TAGS = {
    'exp4': 'Experiment 4  ·  student and LLM',
    'exp3': 'Experiment 3  ·  three students',
}

# ---------------------------------------------------------------- typography
# One data unit = one body line, so every height below is in lines and the
# page size follows from the content (see main()).
F_BODY, F_SMALL, F_VIG, F_TITLE, F_LETTER = 8.2, 6.4, 7.6, 9.4, 12.5
LINESPACING = 1.36
WRAP = 52
VIGNETTE_WRAP = 68
LINE_H = 1.0
PAD = 0.40             # vertical padding inside a bubble, in lines
PADX = 0.022           # horizontal padding inside a bubble, in panel widths
GAP = 0.45             # gap between bubbles
CHAR_W = 0.0138        # panel widths per character at F_BODY (conservative)
AV_IN = 0.24           # avatar diameter, inches
HEADER_H = 2.1         # header bar height, lines
QCARD_LINE = 0.90      # vignette line height, in body lines
DIVIDER_H = 1.3
OUT_ROW = 1.15         # one outcome row, lines
OUT_PAD = 0.5
ARROW = r'$\rightarrow$'   # mathtext: Avenir Next has no U+2192 glyph

HEB2LAT = {'א': 'A', 'ב': 'B', 'ג': 'C', 'ד': 'D'}

# Set in main() once the page geometry is known: inches per x-unit and per
# y-unit, so avatars can be drawn as true circles.
SCALE = [1.0, 1.0]


PANELS = [
    dict(
        kind='pair', source='exp4', row=145, letter='A',
        title='Student defers to a confident LLM',
        verdict='Correct switch', verdict_ok=True,
        vignette=('6-year-old girl: fever, ataxia, malaise. MRI shows widespread '
                  'white-matter lesions. Diagnosis and treatment?\n'
                  '(A) Krabbe disease — supportive care   (B) ADEM — high-dose steroids\n'
                  '(C) MS — high-dose steroids   (D) Viral encephalitis — fluids, rest'),
        turns=[
            ('llm', '[...] MRI findings showing widespread white-matter damage point '
                    'to ADEM. The recommended treatment is high-dose steroids. So the '
                    'correct answer is (B).',
             'לגבי שאלה 25: ... התשובה הנכונה היא (B) ADEM'),
            ('human', "Isn't ADEM both white and grey matter?",
             'Adem זה לא גם לבן וגם אפור?'),
            ('llm', 'Yes, correct, ADEM usually involves both the grey matter and the '
                    'white matter. However [...]',
             'כן, צודק, ADEM בדרך כלל מערב גם את החומר האפור וגם את החומר הלבן ...'),
        ],
    ),
    dict(
        kind='pair', source='exp4', row=34, letter='B',
        title='LLM defers to the student',
        verdict='LLM corrected', verdict_ok=True,
        vignette=('3-month-old girl in winter: rhinorrhoea, fever, respiratory distress. '
                  'Saturation 85%, diffuse wheezing and crackles. Treatment?\n'
                  '(A) Oxygen, steroids, Ventolin   (B) Oxygen, steroids, azithromycin\n'
                  '(C) Oxygen and GC   (D) Oxygen'),
        turns=[
            ('llm', '[...] consistent with RSV. The appropriate treatment is (A) '
                    'Oxygen, steroids and Ventolin inhalation [...]',
             'לגבי שאלה 22, ... התשובה (A)'),
            ('human', "But you don't give Ventolin under one year of age.",
             'אבל לא נותנים ונטולין מתחת לגיל שנה'),
            ('llm', "You're right, I made a mistake [...]. In this case the correct "
                    'option is (D) Oxygen only [...]',
             'צודקת, טעיתי בתגובתי הקודמת ... האופציה הנכונה היא (D) Oxygen בלבד'),
        ],
    ),
    dict(
        kind='pair', source='exp4', row=378, letter='C',
        title='Student defers to a confident LLM',
        verdict='Harmful switch', verdict_ok=False,
        vignette=('TSS with Staphylococcus aureus growth in blood. How should it be '
                  'treated?\n'
                  '(A) Antibiotic against staphylococci   (B) 1st-gen cephalosporin IV\n'
                  '(C) 1st-gen cephalosporin + another antibiotic for blood toxins\n'
                  '(D) 1st-gen cephalosporin + another for dual coverage'),
        turns=[
            ('llm', 'The answer is (A) Antibiotic against staphylococci [...]',
             'לשאלה 6, התשובה היא (A) Antibiotic against staphylococci ...'),
            ('human', 'Why not answer C — first-generation cephalosporins with an '
                      'additional antibiotic for the toxins?',
             'למה לא תשובה ג- צפלוספורינים דור 1 עם אנטיביוטיקה נוספת לטוקסינים?'),
            ('llm', 'Answer C sounds reasonable, but [...] first-generation cephalosporins '
                    'are less effective against MRSA [...]',
             'תשובה ג נשמעת הגיונית, אך ... פחות יעילות נגד MRSA'),
            ('human', "Doesn't a first-generation cephalosporin cover Staph aureus?",
             'צפלוספורין דור 1 לא מכסה סטאפ אאורוס?'),
            ('llm', 'Correct [...] (MSSA). However, in TSS the bacterium may be MRSA, '
                    'which does not respond to them [...]',
             'נכון, צפלוספורינים מדור ראשון ... יעילים נגד MSSA. אבל ... MRSA'),
            ('human', "OK, I'm convinced.", 'בסדר השתכנעתי'),
        ],
    ),
    dict(
        kind='trio', source='exp3', group='E3-G20i', question=38, letter='D',
        title='Dissenter converts the majority',
        verdict='Majority corrected', verdict_ok=True,
        vignette=('65-year-old man, no known underlying disease, presents with a red and '
                  'swollen big toe for the first time in his life. He rates the pain '
                  '10/10. How will you manage this case?\n'
                  '(A) NSAIDs   (B) Allopurinol'),
        turns=[
            ('p3', 'A.', 'א'),
            ('p2', 'Gout, and B? I honestly have no idea..',
             "גאוט וב'? אין לי באמת מושג.."),
            ('p3', 'In an acute attack you give NSAIDs.',
             'בהתקף נותנים NSAIDS'),
            ('p2', 'Interesting — why A?', "מעניין, למה א'?"),
            ('p1', 'Gout, but acute — NSAIDs, indeed.',
             'גאוט אבל אקוטית nsaids באמת'),
            ('p2', 'Got it, and I learned something.', 'קיבלתי ולמדתי'),
        ],
    ),
]


# ---------------------------------------------------------------- data
def load_pair_annotation(df, row):
    r = df.loc[row]
    return dict(kind='pair', target=r['doc.target'], rows=[
        ('human', r['doc.init_question_answer_mapped'], r['doc.final_question_answer_mapped'],
         int(r['doc.init_question_confidence']), int(r['doc.final_question_confidence'])),
        ('llm', r['doc.first_bot_answer_mapped'], r['filtered_resps'],
         int(math.ceil(r['doc.answer_confidence_level'] / 2)), None),
    ])


def load_trio_annotation(df3, group, question):
    sub = df3[(df3.control == 0) & (df3.group_num == group)
              & (df3.question_number == question)].sort_values('subject')
    assert len(sub) == 3, f'expected 3 participants, got {len(sub)}'
    rows = []
    for i, (_, r) in enumerate(sub.iterrows(), start=1):
        rows.append((f'p{i}', HEB2LAT[r['ans_start']], HEB2LAT[r['ans_end']],
                     int(r['conf_val_start']), int(r['conf_val_end'])))
    correct = sub[sub.ans_start_correct == 1]
    target = HEB2LAT[correct.iloc[0]['ans_start']] if len(correct) else \
        HEB2LAT[sub[sub.ans_end_correct == 1].iloc[0]['ans_end']]
    return dict(kind='trio', target=target, rows=rows)


# ---------------------------------------------------------------- geometry
def vignette_lines(panel):
    out = []
    for para in panel['vignette'].split('\n'):
        out.extend(textwrap.wrap(para, VIGNETTE_WRAP))
    return out


def _qcard_h(panel):
    # label row + text lines + answer row
    return 0.9 + len(vignette_lines(panel)) * QCARD_LINE + 1.1 + 2 * PAD


def _bubble_geom(speaker, text):
    _, _, _, _, right = SPEAKERS[speaker]
    lines = textwrap.wrap(text, WRAP)
    name_h = 0.0 if speaker in ('llm', 'human') else 0.85
    height = len(lines) * LINE_H + name_h + 2 * PAD
    width = min(0.80, max(0.18, max(len(l) for l in lines) * CHAR_W + 2 * PADX))
    return lines, height, width, name_h


def _outcome_h(ann):
    return len(ann['rows']) * OUT_ROW + 2 * OUT_PAD


def measure(panel, ann):
    units = HEADER_H + 0.6 + _qcard_h(panel) + DIVIDER_H
    for speaker, text, _ in panel['turns']:
        units += _bubble_geom(speaker, text)[1] + GAP
    return units + _outcome_h(ann) + 0.7


# ---------------------------------------------------------------- icons
def avatar(ax, x, y_center, kind, fill, accent, size_in=AV_IN):
    """A round avatar of diameter size_in inches, drawn in an equal-aspect inset."""
    w, h = size_in / SCALE[0], size_in / SCALE[1]
    ins = ax.inset_axes([x, y_center - h / 2, w, h], transform=ax.transData)
    ins.set_xlim(0, 1)
    ins.set_ylim(0, 1)
    ins.set_aspect('equal')
    ins.axis('off')
    ins.patch.set_alpha(0)
    disc = Circle((0.5, 0.5), 0.5, facecolor=accent, edgecolor='none')
    ins.add_patch(disc)
    if kind == 'robot':
        ins.add_patch(FancyBboxPatch((0.26, 0.28), 0.48, 0.38,
                                     boxstyle='round,pad=0,rounding_size=0.09',
                                     facecolor='white', edgecolor='none'))
        for ex in (0.40, 0.60):
            ins.add_patch(Circle((ex, 0.50), 0.055, facecolor=accent, edgecolor='none'))
        ins.plot([0.40, 0.60], [0.37, 0.37], color=accent, lw=0.9, solid_capstyle='round')
        ins.plot([0.5, 0.5], [0.66, 0.77], color='white', lw=1.1)
        ins.add_patch(Circle((0.5, 0.80), 0.045, facecolor='white', edgecolor='none'))
    else:
        ins.add_patch(Circle((0.5, 0.62), 0.17, facecolor='white', edgecolor='none'))
        body = Wedge((0.5, 0.05), 0.36, 0, 180, facecolor='white', edgecolor='none')
        body.set_clip_path(disc)
        ins.add_patch(body)
    return w, h


# ---------------------------------------------------------------- drawing
def draw_header(ax, y_top, panel):
    ok = panel['verdict_ok']
    ax.add_patch(FancyBboxPatch(
        (0, y_top - HEADER_H), 1.0, HEADER_H,
        boxstyle='round,pad=0,rounding_size=0.025',
        facecolor=C_HEADER, edgecolor='none', clip_on=False))
    # square off the bottom edge so the card continues seamlessly
    ax.add_patch(FancyBboxPatch((0, y_top - HEADER_H), 1.0, 0.6, boxstyle='square,pad=0',
                                facecolor=C_HEADER, edgecolor='none', clip_on=False))
    ax.text(0.03, y_top - HEADER_H / 2, panel['letter'], fontsize=F_LETTER,
            fontweight='bold', color='white', va='center')
    ax.text(0.085, y_top - HEADER_H / 2, panel['title'], fontsize=F_TITLE,
            fontweight='bold', color='white', va='center')
    ax.text(0.975, y_top - HEADER_H / 2, panel['verdict'].upper(), fontsize=6.0,
            fontweight='bold', color='white', va='center', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=C_OK if ok else C_BAD,
                      edgecolor='none'))


def draw_qcard(ax, y_top, panel, target):
    h = _qcard_h(panel)
    ax.add_patch(FancyBboxPatch(
        (0.03, y_top - h), 0.94, h, boxstyle='round,pad=0.004,rounding_size=0.02',
        facecolor=C_QCARD, edgecolor='#D6D9DE', linewidth=0.6, clip_on=False))
    ax.add_patch(FancyBboxPatch((0.03, y_top - h), 0.012, h, boxstyle='square,pad=0',
                                facecolor=C_HEADER, edgecolor='none', clip_on=False))
    ax.text(0.065, y_top - PAD - 0.05, 'QUESTION', fontsize=6.0, fontweight='bold',
            color=C_MUTED, va='top')
    ax.text(0.065, y_top - PAD - 0.95, '\n'.join(vignette_lines(panel)), va='top',
            ha='left', fontsize=F_VIG, color=C_INK, linespacing=1.28)
    ax.text(0.945, y_top - h + PAD + 0.12, f'correct answer  {target}', fontsize=6.6,
            fontweight='bold', color=C_OK, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=C_OK_BG, edgecolor='none'))
    return y_top - h


def draw_divider(ax, y_top):
    y = y_top - DIVIDER_H / 2
    ax.plot([0.06, 0.40], [y, y], color='#CDD1D6', lw=0.6)
    ax.plot([0.60, 0.94], [y, y], color='#CDD1D6', lw=0.6)
    ax.text(0.5, y, 'DISCUSSION', fontsize=6.0, fontweight='bold', color=C_MUTED,
            va='center', ha='center')
    return y_top - DIVIDER_H


def draw_bubble(ax, y_top, speaker, text):
    name, kind, fill, accent, right = SPEAKERS[speaker]
    lines, height, width, name_h = _bubble_geom(speaker, text)
    av_w = AV_IN / SCALE[0]
    x = (0.97 - av_w - 0.018 - width) if right else (0.03 + av_w + 0.018)
    y_bot = y_top - height
    ax.add_patch(FancyBboxPatch(
        (x, y_bot), width, height, boxstyle='round,pad=0.004,rounding_size=0.028',
        facecolor=fill, edgecolor='none', clip_on=False))
    # tail toward the avatar
    if right:
        tail = [(x + width, y_bot + 0.55), (x + width + 0.022, y_bot + 0.15),
                (x + width - 0.01, y_bot + 0.12)]
    else:
        tail = [(x, y_bot + 0.55), (x - 0.022, y_bot + 0.15), (x + 0.01, y_bot + 0.12)]
    ax.add_patch(Polygon(tail, closed=True, facecolor=fill, edgecolor='none', clip_on=False))
    av_x = (0.97 - av_w) if right else 0.03
    avatar(ax, av_x, y_bot + (AV_IN / SCALE[1]) / 2 + 0.05, kind, fill, accent)

    ty = y_top - PAD - 0.05
    if name_h:
        ax.text(x + PADX, ty, name, fontsize=F_SMALL, fontweight='bold', color=accent,
                va='top')
        ty -= name_h
    ax.text(x + PADX, ty, '\n'.join(lines), va='top', ha='left', fontsize=F_BODY,
            color=C_INK, linespacing=LINESPACING)
    return y_bot


def draw_outcome(ax, y_top, ann):
    h = _outcome_h(ann)
    ax.add_patch(FancyBboxPatch(
        (0.14, y_top - h), 0.72, h, boxstyle='round,pad=0.004,rounding_size=0.03',
        facecolor=C_SYS, edgecolor='none', clip_on=False))
    y = y_top - OUT_PAD - OUT_ROW / 2
    for key, pre, post, cpre, cpost in ann['rows']:
        name, kind, fill, accent, _ = SPEAKERS[key]
        avatar(ax, 0.17, y, kind, fill, accent, size_in=0.17)
        ax.text(0.235, y, name, fontsize=F_SMALL + 0.4, fontweight='bold', color=C_INK,
                va='center')
        ax.text(0.50, y, pre, fontsize=F_BODY, fontweight='bold', va='center', ha='center',
                color=C_OK if pre == ann['target'] else C_BAD)
        ax.text(0.56, y, ARROW, fontsize=7, color=C_MUTED, va='center', ha='center')
        ax.text(0.62, y, post, fontsize=F_BODY, fontweight='bold', va='center', ha='center',
                color=C_OK if post == ann['target'] else C_BAD)
        conf = f'{cpre}' if cpost is None else f'{cpre} {ARROW} {cpost}'
        ax.text(0.70, y, f'confidence {conf}', fontsize=F_SMALL, color=C_MUTED,
                va='center')
        y -= OUT_ROW


def draw_panel(ax, panel, ann, y_span):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_span)
    ax.axis('off')
    # chat window: wallpaper card
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1.0, y_span, boxstyle='round,pad=0,rounding_size=0.025',
        facecolor=C_WALL, edgecolor='#D6D9DE', linewidth=0.6, clip_on=False))
    y = y_span
    draw_header(ax, y, panel)
    y -= HEADER_H + 0.6
    y = draw_qcard(ax, y, panel, ann['target'])
    y = draw_divider(ax, y)
    for speaker, text, _heb in panel['turns']:
        y = draw_bubble(ax, y, speaker, text) - GAP
    draw_outcome(ax, _outcome_h(ann) + 0.7, ann)


def draw_legend(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2.2)
    ax.axis('off')
    y = 1.55
    x = 0.0
    for key, label in (('llm', 'LLM'), ('human', 'Student  ·  Experiment 4 (A–C)')):
        name, kind, fill, accent, _ = SPEAKERS[key]
        w, _ = avatar(ax, x, y, kind, fill, accent, size_in=0.17)
        ax.text(x + w + 0.006, y, label, fontsize=7.0, color=C_INK, va='center')
        x += w + 0.006 + 0.012 * len(label) + 0.02
    for key in ('p1', 'p2', 'p3'):
        name, kind, fill, accent, _ = SPEAKERS[key]
        w, _ = avatar(ax, x, y, kind, fill, accent, size_in=0.17)
        x += w + 0.003
    ax.text(x + 0.004, y, 'Participants 1–3  ·  Experiment 3 (D)', fontsize=7.0, color=C_INK,
            va='center')
    x += 0.30
    ax.text(x, y, 'A', fontsize=8, fontweight='bold', color=C_OK, va='center')
    ax.text(x + 0.013, y, 'correct answer', fontsize=7.0, color=C_MUTED, va='center')
    ax.text(x + 0.125, y, 'A', fontsize=8, fontweight='bold', color=C_BAD, va='center')
    ax.text(x + 0.138, y, 'incorrect', fontsize=7.0, color=C_MUTED, va='center')
    ax.text(0.0, 0.5, 'Outcome line: answers before ' + ARROW + ' after; human confidence before ' + ARROW + ' after (1–5); '
                      'LLM initial confidence (0–5). [...] marks abridged text.',
            fontsize=6.8, color=C_MUTED, va='center')


# ---------------------------------------------------------------- caption
def pattern_counts(df4, df3):
    """How often each panel's pattern occurs, so the caption can say."""
    h_pre = df4['user_before_correct'].astype(bool)
    h_post = df4['user_after_correct'].astype(bool)
    l_pre = df4['bot_before_correct'].astype(bool)
    l_post = df4['filtered_resps'] == df4['doc.target']

    splits = converted = narrow = narrow_full = 0
    iv = df3[df3.control == 0]
    for _, sub in iv.groupby(['group_num', 'question_number']):
        if len(sub) != 3:
            continue
        vc = sub.ans_start.value_counts()
        if len(vc) != 2 or vc.iloc[0] != 2:
            continue
        rebel = sub[sub.ans_start == vc.index[1]].iloc[0]
        coalition = sub[sub.ans_start == vc.index[0]]
        if not (rebel.ans_start_correct == 1 and coalition.ans_start_correct.sum() == 0):
            continue
        splits += 1
        full = coalition.ans_end_correct.sum() == 2
        converted += bool(full)
        if rebel.conf_val_start > coalition.conf_val_start.max():
            narrow += 1
            narrow_full += bool(full)

    return dict(A=int((~h_pre & l_pre & h_post).sum()),
                B=int((~l_pre & h_pre & l_post).sum()),
                C=int((h_pre & ~l_pre & ~h_post).sum()),
                total4=len(df4), D=converted, splits3=splits,
                D_conf=narrow_full, splits3_conf=narrow)


CAPTION = 'Figure 4. Knowledge-certainty diversity inside a single conversation. Four example exchanges, translated from Hebrew and shown as they unfolded. A–C come from Experiment 4 (a student and an LLM), D from Experiment 3 (three students). Each panel shows the question put to all agents, with its correct answer, above the discussion; the line at the bottom gives each agent\'s answer before and after the discussion (green correct, red incorrect) and human confidence before and after discussion, and the LLM’s initial self-consistency confidence. Confidence is self-reported on 1–5 by students; for the LLM it is on a 0–5 scale based on 10 self-consistency runs (Methods). Excerpts were selected to illustrate the patterns described below, not drawn at random; the full conversation corpus is released alongside publication for deeper analysis of its content (Data availability). (A) A student who is unsure and wrong challenges the LLM; it concedes the point but keeps its answer, and the student adopts it and is right. (B) The LLM is wrong at low confidence (level 1); the student supplies a paediatric prescribing rule and it revises to the correct answer. (C) The mirror image of A: the student is right, the maximally confident LLM is wrong, and the student abandons a correct answer while growing more confident in it. (D) In a trio, the lone dissenter is both the most confident member and the only one who is right, and brings the whole majority across; no one states a confidence number, the least confident participant simply says "I honestly have no idea." Panels A and C illustrate that superficially similar patterns of challenge and deference can lead to different accuracy outcomes. Long LLM turns are abridged at [...] and C omits a closing courtesy line; no claim or conclusion is removed. Participants are numbered, never named.'


# ---------------------------------------------------------------- main
def main():
    df4 = pd.read_csv(DATA4)
    df3 = pd.read_csv(DATA3)

    anns = []
    for p in PANELS:
        if p['kind'] == 'pair':
            anns.append(load_pair_annotation(df4, p['row']))
        else:
            anns.append(load_trio_annotation(df3, p['group'], p['question']))

    counts = pattern_counts(df4, df3)

    # Lay the figure out from the content rather than guessing a size: one
    # data unit equals one body line on the page.
    IPU = F_BODY * LINESPACING / 72 * 1.02   # inches per body line
    M_TOP, M_BOT, ROW_GAP = 0.08, 0.05, 0.28
    LEGEND_H, LEGEND_GAP = 0.42, 0.16
    FIG_W = 8.6
    COL_X, COL_W, COL_GAP = 0.03, 0.463, 0.044
    SCALE[0] = COL_W * FIG_W      # inches per x-unit
    SCALE[1] = IPU                # inches per y-unit

    spans = [measure(p, a) for p, a in zip(PANELS, anns)]
    n_rows = (len(PANELS) + 1) // 2
    row_span = [max(spans[r * 2:r * 2 + 2]) for r in range(n_rows)]
    row_h = [s * IPU for s in row_span]
    fig_h = (M_TOP + sum(row_h) + ROW_GAP * (n_rows - 1)
             + LEGEND_GAP + LEGEND_H + M_BOT)

    fig = plt.figure(figsize=(FIG_W, fig_h))
    for i, (panel, ann) in enumerate(zip(PANELS, anns)):
        col, row = i % 2, i // 2
        top_in = M_TOP + sum(row_h[:row]) + ROW_GAP * row
        ax_h_in = row_span[row] * IPU
        bottom_frac = (fig_h - top_in - ax_h_in) / fig_h
        ax = fig.add_axes([COL_X + col * (COL_W + COL_GAP), bottom_frac,
                           COL_W, ax_h_in / fig_h])
        draw_panel(ax, panel, ann, row_span[row])

    lg = fig.add_axes([COL_X, M_BOT / fig_h, 2 * COL_W + COL_GAP, LEGEND_H / fig_h])
    SCALE_LEG = [(2 * COL_W + COL_GAP) * FIG_W, LEGEND_H / 2.2]
    saved = SCALE[:]
    SCALE[0], SCALE[1] = SCALE_LEG
    draw_legend(lg)
    SCALE[0], SCALE[1] = saved

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'figure_conversations.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Counts (190/39/27 of 562; 15/33 for D) no longer appear in the caption
    # (a reader found them more confusing than helpful) but pattern_counts()
    # is still computed and printed to console below for provenance.
    caption = CAPTION
    cap_path = os.path.join(OUT_DIR, 'figure_conversations_caption.md')
    bits = [f"{p['letter']} = row {p['row']}" if p['kind'] == 'pair'
            else f"{p['letter']} = group {p['group']}, question {p['question']}"
            for p in PANELS]
    with open(cap_path, 'w') as f:
        f.write(caption + '\n\nSources: ' + '; '.join(bits)
                + '. Experiment 4 panels from medcom2_bot_post_conversation_updated.csv; '
                  'Experiment 3 panel from medcom1_fulldata.csv (answers and confidence) '
                  'and data/conversations/experiment3.csv (transcript).\n')

    print(f'written to {out}')
    print(f'caption written to {cap_path}')
    for panel, ann in zip(PANELS, anns):
        who = f"row {panel['row']}" if panel['kind'] == 'pair' else \
            f"group {panel['group']} Q{panel['question']}"
        detail = '; '.join(f"{SPEAKERS[k][0]} {pre}->{post} (conf {cpre}"
                           + (f"->{cpost})" if cpost is not None else ")")
                           for k, pre, post, cpre, cpost in ann['rows'])
        print(f"  {panel['letter']} ({who}): {detail}; correct {ann['target']}")
    print(f"  Exp 4 patterns of {counts['total4']} discussed items: "
          f"A={counts['A']}, B={counts['B']}, C={counts['C']}")
    print(f"  Exp 3: {counts['D']} of {counts['splits3']} lone-correct-dissenter items "
          f"ended with the whole majority corrected; of the {counts['splits3_conf']} where "
          f"that dissenter was also the most confident, {counts['D_conf']} did")


if __name__ == '__main__':
    main()
