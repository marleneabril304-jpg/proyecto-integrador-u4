import duckdb
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Optimización Agrícola", layout="wide")
sns.set_theme(style="whitegrid")


# ---------------------------------------------------------
# HITO 1: Telemetría con DuckDB (semilla fija 0.42)
# ---------------------------------------------------------
@st.cache_data
def cargar_datos() -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    con.execute("SELECT setseed(0.42)")  # Paso 1.1

    # Paso 1.2: tabla de N = 100 parcelas
    con.execute("""
        CREATE TABLE telemetria AS
        WITH base AS (
            SELECT
                i AS parcela_id,
                ROUND(100.0 + random() * 200.0, 2) AS volumen_riego,
                ROUND(50.0 + random() * 100.0, 2)  AS fertilizante_kg,
                random()                           AS p_trat,
                (random() * 1.6) - 0.8             AS ruido,
                ROUND(18.0 + random() * 14.0, 1)   AS temperatura_c
            FROM range(1, 101) AS t(i)
        )
        SELECT
            parcela_id,
            volumen_riego,
            fertilizante_kg,
            CASE WHEN p_trat < 0.50 THEN 'Tratamiento_A'
                 ELSE 'Tratamiento_B' END AS tipo_tratamiento,
            ROUND(
                5.0
                + 0.035 * volumen_riego
                + 0.020 * fertilizante_kg
                + CASE WHEN p_trat >= 0.50 THEN 1.8 ELSE 0.0 END
                + ruido
            , 2) AS rendimiento_ton,
            temperatura_c
        FROM base
    """)

    # Paso 1.3: limpieza / extracción SQL antes de pasar a Python
    df = con.execute("""
        SELECT parcela_id, volumen_riego, fertilizante_kg,
               tipo_tratamiento, rendimiento_ton, temperatura_c
        FROM telemetria
        WHERE volumen_riego IS NOT NULL
          AND fertilizante_kg IS NOT NULL
          AND rendimiento_ton IS NOT NULL
        ORDER BY parcela_id
    """).df()
    con.close()
    return df


df_total = cargar_datos()

# ---------------------------------------------------------
# HITO 4 (4.2): filtros dinámicos en la barra lateral
# ---------------------------------------------------------
st.sidebar.header("Filtros")
trat = st.sidebar.multiselect(
    "Tipo de tratamiento",
    ["Tratamiento_A", "Tratamiento_B"],
    default=["Tratamiento_A", "Tratamiento_B"],
)
riego_min, riego_max = st.sidebar.slider(
    "Volumen de riego (m³/ha)", 100.0, 300.0, (100.0, 300.0)
)

df = df_total[
    df_total["tipo_tratamiento"].isin(trat)
    & df_total["volumen_riego"].between(riego_min, riego_max)
].copy()

st.title("Optimización Inferencial del Rendimiento Agrícola y Eficiencia de Riego")
st.caption("Invernaderos de jitomate/aguacate · Análisis y Exploración de Datos (INC-2505)")

if len(df) < 10:
    st.warning("Hay muy pocos datos con estos filtros. Amplía la selección.")
    st.stop()

# ---------------------------------------------------------
# HITO 2: Correlación y regresión OLS
# ---------------------------------------------------------
corr = df[["volumen_riego", "fertilizante_kg", "temperatura_c", "rendimiento_ton"]].corr(
    method="pearson"
)

X = sm.add_constant(df[["volumen_riego", "fertilizante_kg"]])
modelo = sm.OLS(df["rendimiento_ton"], X).fit()
r2_aj = modelo.rsquared_adj
rmse = float(np.sqrt(np.mean(modelo.resid ** 2)))

# ---------------------------------------------------------
# HITO 3: t-Student de dos muestras independientes (cola derecha)
# ---------------------------------------------------------
a = df.loc[df["tipo_tratamiento"] == "Tratamiento_A", "rendimiento_ton"]
b = df.loc[df["tipo_tratamiento"] == "Tratamiento_B", "rendimiento_ton"]
hay_dos_grupos = len(a) > 1 and len(b) > 1

if hay_dos_grupos:
    t_stat, p_valor = stats.ttest_ind(b, a, equal_var=True, alternative="greater")
    dif = b.mean() - a.mean()
    gl = len(a) + len(b) - 2
    sp2 = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / gl
    ee = np.sqrt(sp2 * (1 / len(a) + 1 / len(b)))
    t_crit = stats.t.ppf(0.975, gl)
    ic_inf, ic_sup = dif - t_crit * ee, dif + t_crit * ee

# ---------------------------------------------------------
# HITO 4 (4.1): tarjetas KPI
# ---------------------------------------------------------
k1, k2, k3 = st.columns(3)
k1.metric("Rendimiento promedio global", f"{df['rendimiento_ton'].mean():.2f} ton/ha")
k2.metric("R² ajustado del modelo", f"{r2_aj * 100:.2f} %")
if hay_dos_grupos:
    k3.metric("p-valor (prueba t)", f"{p_valor:.3g}")
else:
    k3.metric("p-valor (prueba t)", "Necesita A y B")

st.divider()

# ---------------------------------------------------------
# Cuadrícula 2x2 con las 4 gráficas
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Mapa de calor de correlación (Pearson)")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", vmin=-1, vmax=1, ax=ax)
    ax.set_xticklabels(["Riego", "Fertilizante", "Temperatura", "Rendimiento"], rotation=30)
    ax.set_yticklabels(["Riego", "Fertilizante", "Temperatura", "Rendimiento"], rotation=0)
    st.pyplot(fig)

with col2:
    st.subheader("2. Dispersión con recta de regresión OLS")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.regplot(
        data=df, x="volumen_riego", y="rendimiento_ton", ci=95,
        scatter_kws={"alpha": 0.6}, line_kws={"color": "red"}, ax=ax,
    )
    ax.set_xlabel("Volumen de riego (m³/ha)")
    ax.set_ylabel("Rendimiento (ton/ha)")
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    st.subheader("3. Boxplot comparativo de tratamientos")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.boxplot(data=df, x="tipo_tratamiento", y="rendimiento_ton",
                order=["Tratamiento_A", "Tratamiento_B"], ax=ax)
    medias = df.groupby("tipo_tratamiento")["rendimiento_ton"].mean()
    for i, nombre in enumerate(["Tratamiento_A", "Tratamiento_B"]):
        if nombre in medias:
            ax.scatter(i, medias[nombre], color="black", marker="D", zorder=5)
    ax.set_xlabel("Tipo de tratamiento (A vs B)")
    ax.set_ylabel("Rendimiento (ton/ha)")
    st.pyplot(fig)

with col4:
    st.subheader("4. Histograma de residuos del modelo")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.histplot(modelo.resid, kde=True, stat="density", bins=15, ax=ax)
    ax.set_xlabel("Residuo (ton/ha)")
    ax.set_ylabel("Densidad")
    st.pyplot(fig)

st.divider()

# ---------------------------------------------------------
# Resultados e interpretación
# ---------------------------------------------------------
st.header("Resultados del análisis")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Modelo de regresión múltiple (OLS)")
    st.markdown(
        f"**Rendimiento = {modelo.params['const']:.3f} "
        f"+ {modelo.params['volumen_riego']:.4f}·Riego "
        f"+ {modelo.params['fertilizante_kg']:.4f}·Fertilizante**"
    )
    st.write(f"R² ajustado = {r2_aj:.4f} | RMSE = {rmse:.3f} ton/ha")
    tabla = pd.DataFrame({"Coeficiente": modelo.params, "p-valor": modelo.pvalues})
    st.dataframe(tabla.style.format({"Coeficiente": "{:.4f}", "p-valor": "{:.3g}"}))

with c2:
    st.subheader("Prueba t (H₀: μB − μA ≤ 0 vs H₁: μB − μA > 0)")
    if hay_dos_grupos:
        st.write(f"Media A = {a.mean():.2f} | Media B = {b.mean():.2f}")
        st.write(f"Diferencia (B − A) = {dif:.2f} ton/ha")
        st.write(f"Estadístico t = {t_stat:.3f} | p-valor = {p_valor:.3g}")
        st.write(f"IC 95% de la diferencia: [{ic_inf:.2f}, {ic_sup:.2f}] ton/ha")
        if p_valor < 0.05:
            st.success("Se RECHAZA H₀: el Tratamiento B incrementa el rendimiento. "
                       "Se recomienda autorizar la inversión.")
        else:
            st.error("NO se rechaza H₀: no hay evidencia suficiente de que B sea mejor.")
    else:
        st.info("Selecciona ambos tratamientos para ver la prueba t.")

with st.expander("Ver tabla de datos (telemetría)"):
    st.dataframe(df, width="stretch")
