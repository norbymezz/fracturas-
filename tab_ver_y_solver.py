# === Nueva estructura de pestañas: Verificación aislada + Conexión al solver ===
import streamlit as st
import pandas as pd
import numpy as np

# Tabs principales
tab_verif, tab_solver = st.tabs(["✅ Verificación aislada", "🧮 Conexión Solver"])

# ---------------- VERIFICACIÓN AISLADA -----------------
with tab_verif:
    st.subheader("Verificación aislada contra curva del paper")

    n_ref = st.number_input("N puntos de referencia", 1, 200, 20)

    def default_verif_table(n):
        return pd.DataFrame({"t_días": np.linspace(0.001, 10, n), "p_psi_ref": np.linspace(5, 25, n)})

    verif_df = st.data_editor(default_verif_table(n_ref), num_rows="dynamic", key="verif_table")

    # Curva modelo temporal
    t = verif_df["t_días"].values
    p_ref = verif_df["p_psi_ref"].values
    p_mod = 5 + 2 * np.sqrt(t)

    err_rel = np.mean(np.abs((p_mod - p_ref) / np.maximum(p_ref, 1e-6)))

    st.metric("Error relativo medio", f"{err_rel:.3e}")

    if err_rel < 1e-3:
        st.success(f"✅ Validado (tol = 1e-3)")
    else:
        st.warning(f"⚠️ No validado (err = {err_rel:.3e})")

    df_comp = pd.DataFrame({
        "t_días": t,
        "p_ref (psi)": p_ref,
        "p_model (psi)": p_mod,
        "error_abs (psi)": np.abs(p_mod - p_ref)
    })

    st.dataframe(df_comp, use_container_width=True)

    csv = df_comp.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Descargar CSV comparativo", csv, "comparacion_verif_aislada.csv", "text/csv")

# ---------------- CONEXIÓN AL SOLVER -----------------
with tab_solver:
    st.subheader("Comparación directa con resultados del solver")

    st.markdown(
        "Esta pestaña toma el vector de presiones calculado por el solver real y lo compara con la tabla de referencia."
    )

    # Simulación: en la app final se obtendrá del solver (por ejemplo st.session_state['Pwf'] o matriz P)
    if "solver_output" not in st.session_state:
        # Generamos un ejemplo ficticio
        t_solver = np.linspace(0.001, 10, 20)
        Pwf_solver = 5 + 1.8 * np.sqrt(t_solver)
        st.session_state.solver_output = pd.DataFrame({"t_días": t_solver, "Pwf_model (psi)": Pwf_solver})

    solver_df = st.session_state.solver_output
    st.write("### Datos del solver (simulados):")
    st.dataframe(solver_df, use_container_width=True)

    # Cargar curva de referencia
    st.write("### Curva de referencia")
    ref_upload = st.file_uploader("Subí tu CSV de referencia (t_días, p_ref)", type=["csv"])

    if ref_upload is not None:
        ref_df = pd.read_csv(ref_upload)
        # Interpolación al dominio del solver
        p_interp = np.interp(solver_df["t_días"], ref_df["t_días"], ref_df["p_ref"])
        err_rel_solver = np.mean(np.abs((solver_df["Pwf_model (psi)"] - p_interp) / np.maximum(p_interp, 1e-6)))

        st.metric("Error relativo medio (solver vs ref)", f"{err_rel_solver:.3e}")
        if err_rel_solver < 1e-3:
            st.success(f"✅ Validado (tol = 1e-3)")
        else:
            st.warning(f"⚠️ No validado (err = {err_rel_solver:.3e})")

        df_merge = pd.DataFrame({
            "t_días": solver_df["t_días"],
            "p_ref (psi)": p_interp,
            "p_solver (psi)": solver_df["Pwf_model (psi)"],
            "error_abs (psi)": np.abs(solver_df["Pwf_model (psi)"] - p_interp)
        })

        st.dataframe(df_merge, use_container_width=True)

        csv2 = df_merge.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Descargar comparación Solver", csv2, "comparacion_solver_vs_ref.csv", "text/csv")
