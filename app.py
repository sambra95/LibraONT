"""LibraONT - Streamlit entry point. One FASTQ is described on its own; several
are followed over time as one selection (``gui.selection``)."""

from __future__ import annotations

import streamlit as st

from gui import inputs, results, runner, selection
from libraont import theme, timecourse  # theme activates the shared Plotly template

st.set_page_config(page_title="LibraONT", page_icon="🧬", layout="wide")

# Light touch of theme colour on the title bar.
st.html(
    f"<h1 style='color:{theme.PALETTE['primary_dark']};margin-bottom:0'>LibraONT</h1>"
    f"<p style='color:{theme.PALETTE['muted']};margin-top:4px'>"
    "Turns raw Oxford Nanopore FASTQ reads into a picture of your library's "
    "quality and composition.</p>")

samples, error, time_unit, run_clicked = inputs.render_sidebar()

if run_clicked:
    if error:
        st.error(error)
    else:
        bar = st.progress(0.0, text="Starting…")

        def step(frac: float, msg: str) -> None:
            bar.progress(frac, text=msg)

        try:
            st.session_state["result"] = (
                runner.run_analysis(samples[0][2], progress=step) if len(samples) == 1
                else selection.run(samples, time_unit, progress=step))
        except Exception as exc:
            st.session_state.pop("result", None)
            st.error(f"Analysis failed: {exc}")
        finally:
            bar.empty()

result = st.session_state.get("result")
if result is None:
    st.info("Upload one FASTQ to describe a library, or several - one per time "
            "point - to follow a selection, then click **Run analysis**.")
elif isinstance(result, timecourse.TimeCourse):
    selection.render(result)
else:
    results.render(result)
