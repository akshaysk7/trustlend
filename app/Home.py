import json
import sys
from datetime import date, datetime
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "app")]

import importlib
import src.decide
importlib.reload(src.decide)
from src.decide import decide
from src.model import CUTOFF
from audit import log
import db as _db
_db.init_db()
from sidebar import render_sidebar
from style import inject_css, sheet_header, card, stamp, tag, check_ruler, MUTED2

st.set_page_config(page_title="TrustLend", layout="wide")
inject_css()
st.logo(str(ROOT / "app" / "logo.svg"))
T = json.load(open(ROOT / "artifacts" / "thresholds.json"))
render_sidebar(T)

FIELDS = {
    "age": ("Age", 35),
    "MonthlyIncome": ("Monthly income", 5000),
    "DebtRatio": ("Debt ratio (monthly debt / income)", 0.3),
    "RevolvingUtilizationOfUnsecuredLines": ("Credit card use (0.3 = 30% of limit)", 0.3),
    "NumberOfOpenCreditLinesAndLoans": ("Open loans and cards", 8),
    "NumberRealEstateLoansOrLines": ("Home loans", 1),
    "NumberOfDependents": ("Dependents", 0),
    "NumberOfTime30-59DaysPastDueNotWorse": ("Times 30-59 days late", 0),
    "NumberOfTime60-89DaysPastDueNotWorse": ("Times 60-89 days late", 0),
    "NumberOfTimes90DaysLate": ("Times 90+ days late", 0)
}
RATIO_FIELDS = {"DebtRatio", "RevolvingUtilizationOfUnsecuredLines"}
CAPTIONS = {
    "Normal": "Typical, low risk", "Borderline": "Right next to the line",
    "Stranger": "Unlike past applicants", "Typo": "Age typed as 420",
    "Deny": "High risk, plain deny",
}

PRESETS = json.load(open(ROOT / "app" / "presets.json"))

for col, (_, default) in FIELDS.items():
    st.session_state.setdefault(col, float(default) if col in RATIO_FIELDS else int(default))
# session_state queue kept for sidebar badge compat; DB is source of truth
st.session_state.setdefault("file_seq", 138)

def load_preset(name):
    for col, v in PRESETS[name].items():
        if col in RATIO_FIELDS:
            st.session_state[col] = float(v) if v is not None else 0.0
        else:
            st.session_state[col] = int(v) if v is not None else 0
    st.session_state["preset_name"] = name

sheet_header(
    "LOAN DETERMINATION · FORM TL-1", "Check an applicant",
    "A loan AI that knows when to ask a human.",
    [("FILE NO.", f"TL-{st.session_state.file_seq:04d}"), ("DATE", date.today().strftime("%d.%m.%y"))],
)

with card("demo", "A · Demo applicants"):
    active = st.session_state.get("preset_name")
    cols = st.columns(len(PRESETS))
    for c, name in zip(cols, PRESETS):
        with c:
            with st.container(key=f"preset_active_{name}" if name == active else f"preset_{name}"):
                st.button(name.upper(), key=f"btn_{name}", on_click=load_preset, args=(name,), use_container_width=True)
            st.caption(CAPTIONS.get(name, ""))

left, right = st.columns([1, 1.4])
with left:
    with card("applicant_data", "B · Applicant data", meta="10 FIELDS"):
        with st.form("applicant"):
            for col, (label, _) in FIELDS.items():
                if col in RATIO_FIELDS:
                    st.number_input(label, min_value=0.0, step=0.01, format="%.2f", key=col)
                elif col == "age":
                    st.number_input(label, min_value=18, step=1, key=col)
                else:
                    st.number_input(label, min_value=0, step=1, key=col)
            submitted = st.form_submit_button("Run determination →")

with right:
    if not submitted and "last_result" not in st.session_state:
        with card("determination", "C · Determination"):
            st.write("Load a demo applicant or fill the form, then run a determination.")
    else:
        if submitted:
            applicant = {col: st.session_state[col] for col in FIELDS}
            r = decide(applicant)
            name = st.session_state.get("preset_name", "manual entry")
            st.session_state.file_seq += 1
            file_no = f"TL-{st.session_state.file_seq:04d}"
            st.session_state["last_result"] = {"name": name, "applicant": applicant, "result": r, "file": file_no}
            if r["decision"] == "REVIEW":
                _db.create_review_case(file_no, name, applicant, r)
            else:
                log(name, r, r["decision"], "AI")

        case = st.session_state["last_result"]
        r, name, file_no = case["result"], case["name"], case["file"]

        with card("determination", "C · Determination", meta=f"{file_no} · {name.upper()}"):
            stcol, datacol = st.columns([1, 1.3])
            with stcol:
                st.markdown("<br>", unsafe_allow_html=True)
                stamp(r["decision"],
                      "TO A HUMAN REVIEWER" if r["decision"] == "REVIEW" else "AUTOMATIC · BOTH CHECKS PASS")
            with datacol:
                trigger = "NONE"
                if r["unsure"] and r["unfamiliar"]:
                    trigger = "CHECK C1 + C2"
                elif r["unsure"]:
                    trigger = "CHECK C1"
                elif r["unfamiliar"]:
                    trigger = "CHECK C2"
                decided_by = "OFFICER · PENDING" if r["decision"] == "REVIEW" else "AI · AUTOMATIC"
                rows = [
                    ("Risk of default", f"{r['p_default']:.1%}"),
                    ("Distance score", f"{r['familiarity_distance']:.2f}"),
                    ("Referred by", trigger),
                    ("Decided by", decided_by),
                ]
                for label, value in rows:
                    st.markdown(
                        f'<div style="display:flex; justify-content:space-between; align-items:baseline; '
                        f'padding:9px 0; border-bottom:1px solid #DDD8CC">'
                        f'<span style="font-size:13px; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; color:{MUTED2}">{label}</span>'
                        f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:18px; font-weight:700">{value}</span></div>',
                        unsafe_allow_html=True,
                    )
            why = r["reasons"][0] if r["reasons"] else "Both checks passed, so the AI decided on its own."
            st.markdown(
                f'<div style="padding-top:14px; margin-top:6px; border-top:1.5px solid #16181B; display:flex; gap:14px">'
                f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:11px; font-weight:700; letter-spacing:0.14em; color:{MUTED2}">WHY</span>'
                f'<span style="font-size:17px; font-weight:600; line-height:1.4">{why}</span></div>',
                unsafe_allow_html=True,
            )

        with card("checks", "D · Safety checks", meta="EITHER FIRES → HUMAN"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div style="display:flex; justify-content:space-between; align-items:center">'
                            f'<span style="font-size:15px; font-weight:700">C1 · Too close to call?</span>'
                            f'{tag("FIRED" if r["unsure"] else "PASS", "fired" if r["unsure"] else "pass")}</div>',
                            unsafe_allow_html=True)
                band = T["band"]
                scale = max(0.40, (CUTOFF + band) * 1.6)
                check_ruler(r["p_default"], scale, CUTOFF - band, CUTOFF + band, CUTOFF,
                            f"{CUTOFF:.1%}", "{:.1%}", r["unsure"])
                st.caption("Inside the zone: the model itself is unsure." if r["unsure"]
                           else "Outside the zone: a clear call.")
            with c2:
                st.markdown(f'<div style="display:flex; justify-content:space-between; align-items:center">'
                            f'<span style="font-size:15px; font-weight:700">C2 · Seen anyone like this?</span>'
                            f'{tag("FIRED" if r["unfamiliar"] else "PASS", "fired" if r["unfamiliar"] else "pass")}</div>',
                            unsafe_allow_html=True)
                fam_t = T["fam_threshold"]
                scale = max(fam_t * 2.5, r["familiarity_distance"] * 1.05, 1)
                check_ruler(r["familiarity_distance"], scale, fam_t, scale, fam_t,
                            f"{fam_t:.2f}", "{:.2f}", r["unfamiliar"])
                st.caption("Past the cut-off: unlike the people the AI learned from." if r["unfamiliar"]
                           else "Under the cut-off: similar people are in the training data.")

        rc1, rc2 = st.columns([1, 1.6])
        with rc1:
            with card("reasons", "E · Reasons"):
                if r["reasons"]:
                    for i, reason in enumerate(r["reasons"], start=1):
                        st.markdown(
                            f'<div style="display:flex; gap:12px; padding:9px 0; border-bottom:1px solid #DDD8CC; font-size:14px">'
                            f'<span style="width:24px; height:20px; flex-shrink:0; display:flex; align-items:center; justify-content:center; '
                            f'font-family:\'JetBrains Mono\',monospace; font-size:11px; font-weight:700; border:1.5px solid #16181B">{i:02d}</span>'
                            f'<span>{reason}</span></div>', unsafe_allow_html=True)
                else:
                    st.write("No flags. Both checks passed cleanly.")
        with rc2:
            with card("neighbours", "F · Comparable past files", meta="C2 SCORE = AVG DISTANCE OF THESE"):
                st.dataframe(pd.DataFrame(r["neighbours"]), hide_index=True, height=280)
