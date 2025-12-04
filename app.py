"""
Lanzador seleccio la app SPE-215031 a ejecutar.
"""

import streamlit as st

def main():
    st.title("SPE-215031 Selector de aplicacin")
    with st.sidebar:
        st.header("Abrir app")
        choice = st.radio("Variante", (
            "SPE-215031 (limpia)",
            "SPE-215031 (UTF-8 completa)",
            "aaapp (interferencia desde cero)",
            "aapwm (paralelos All-in v2)",
        ))
    if choice.startswith("SPE-215031 (limpia)"):
        import app_spe215031_clean as clean_app
        clean_app.app()
        return
    if choice.startswith("SPE-215031 (UTF-8 completa)"):
        import app_spe215031_full_utf8 as full_app
        full_app.app()
        return
    if choice.startswith("aaapp"):
        import aaapp
        st.stop()
    if choice.startswith("aapwm"):
        import aapwm
        st.stop()

if __name__ == "__main__":
    main()
