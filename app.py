"""LibraONT - Streamlit entry point.

Thin GUI shell: collect inputs (``gui.inputs``), run the analysis backend
(``libraont.pipeline``), and render results (``gui.results``). All computation
lives in the ``libraont`` package; this file only wires the UI together.
"""

from __future__ import annotations

import streamlit as st

from libraont import theme  # noqa: F401  (activates the shared Plotly template)
from gui import inputs, results, runner

st.set_page_config(page_title="LibraONT", page_icon="🧬", layout="wide")

# Light touch of theme colour on the title bar.
st.markdown(
    f"<h1 style='color:{theme.PALETTE['primary_dark']};margin-bottom:0'>LibraONT</h1>"
    f"<p style='color:{theme.PALETTE['muted']};margin-top:4px'>"
    "Nanopore mutagenesis-library analysis - orientation, alignment, "
    "base/AA composition and variant diversity.</p>",
    unsafe_allow_html=True,
)

params, run_clicked, error = inputs.render_sidebar()

if run_clicked:
    if error:
        st.error(error)
    else:
        progress = st.progress(0.0, text="Starting…")
        try:
            report = runner.run_analysis(params, progress=lambda f, m: progress.progress(f, text=m))
            st.session_state["report"] = report
        except Exception as exc:
            st.session_state.pop("report", None)
            st.error(f"Analysis failed: {exc}")
        finally:
            progress.empty()

if "report" in st.session_state:
    results.render(st.session_state["report"])
else:
    st.info("Set inputs in the sidebar and click **Run analysis**.")
