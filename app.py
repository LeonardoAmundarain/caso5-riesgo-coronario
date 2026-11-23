"""Prototipo Caso 5: tamizaje de riesgo coronario a 10 años.

Ejecutar:  streamlit run app.py
Si no existe resultados/modelo_chd10a.pkl, la app corre el pipeline una vez.
"""
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src import pipeline as P

st.set_page_config(page_title="Riesgo coronario a 10 años · Caso 5", page_icon="🫀", layout="wide")

AZUL, ROJO, GRIS = "#2563eb", "#dc2626", "#94a3b8"


# --------------------------------------------------------------------------- #
# Carga (con caché)
# --------------------------------------------------------------------------- #
@st.cache_resource
def obtener_modelo():
    ruta = C.DIR_RESULTADOS / C.ARCHIVO_MODELO
    if not ruta.exists():
        P.ejecutar()
    return P.cargar_modelo(ruta)


@st.cache_data
def obtener_reporte():
    return json.loads((C.DIR_RESULTADOS / "reporte_modelo.json").read_text(encoding="utf-8"))


@st.cache_data
def obtener_cohorte():
    return pd.read_csv(C.DIR_DATOS / C.ARCHIVO_COHORTE)


def pct(x, d=1):
    return f"{100 * x:.{d}f} %".replace(".", ",")


def num(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


paquete = obtener_modelo()
reporte = obtener_reporte()
cohorte = obtener_cohorte()
UMBRAL = paquete["umbral"]

# --------------------------------------------------------------------------- #
# Encabezado
# --------------------------------------------------------------------------- #
st.title("🫀 Riesgo coronario a 10 años")
st.caption("Caso 5 · Gestión de Proyectos de Datos (SCY1102) · Regresión logística entrenada con la "
           "cohorte de Framingham. Prototipo académico: no reemplaza una evaluación clínica.")

tab_form, tab_dash, tab_ens = st.tabs(["Evaluar a una persona", "Resultados del modelo", "Contraste con Chile (ENS)"])

# --------------------------------------------------------------------------- #
# 1. Formulario
# --------------------------------------------------------------------------- #
with tab_form:
    col_f, col_r = st.columns([1.1, 1], gap="large")

    with col_f:
        with st.form("persona"):
            st.subheader("Datos del control preventivo")
            a, b = st.columns(2)
            sexo = a.radio("Sexo", ["Hombre", "Mujer"], horizontal=True)
            edad = b.number_input("Edad (años)", *C.RANGOS["AGE"][:3], step=1)

            a, b = st.columns(2)
            sys_ = a.number_input("Presión sistólica (mmHg)", *C.RANGOS["SYSBP"][:3], step=1)
            dia = b.number_input("Presión diastólica (mmHg)", *C.RANGOS["DIABP"][:3], step=1)

            a, b = st.columns(2)
            col = a.number_input("Colesterol total (mg/dL)", *C.RANGOS["TOTCHOL"][:3], step=1)
            glu = b.number_input("Glicemia (mg/dL)", *C.RANGOS["GLUCOSE"][:3], step=1)

            a, b = st.columns(2)
            imc = a.number_input("IMC (kg/m²)", *C.RANGOS["BMI"][:3], step=0.1, format="%.1f")
            fc = b.number_input("Frecuencia cardíaca (lpm)", *C.RANGOS["HEARTRTE"][:3], step=1)

            a, b = st.columns(2)
            fuma = a.toggle("Fumador actual")
            cig = b.number_input("Cigarrillos al día", *C.RANGOS["CIGPDAY"][:3], step=1)

            a, b = st.columns(2)
            diab = a.toggle("Diabetes")
            hta = b.toggle("Usa antihipertensivos")

            enviado = st.form_submit_button("Calcular riesgo", type="primary", use_container_width=True)

    datos = {"AGE": edad, "SEX": 1 if sexo == "Hombre" else 2, "SYSBP": sys_, "DIABP": dia,
             "TOTCHOL": col, "GLUCOSE": glu, "BMI": imc, "HEARTRTE": fc,
             "CURSMOKE": int(fuma), "CIGPDAY": cig if fuma else 0,
             "DIABETES": int(diab), "BPMEDS": int(hta)}

    with col_r:
        st.subheader("Resultado")
        errores = P.validar_persona({**datos, "CIGPDAY": cig})
        if fuma is False and cig > 0:
            st.info("Marcó 0 en fumador: los cigarrillos al día no se consideran.")
            errores = [e for e in errores if "cigarrillos" not in e]
        if errores:
            for e in errores:
                st.error(e)
        elif not enviado and "ya_calculo" not in st.session_state:
            st.write("Complete el formulario y presione **Calcular riesgo**.")
        else:
            st.session_state["ya_calculo"] = True
            riesgo, derivar = P.predecir(paquete, datos)

            m1, m2 = st.columns(2)
            m1.metric("Riesgo estimado a 10 años", pct(riesgo))
            m2.metric("Umbral de derivación", pct(UMBRAL))
            if derivar:
                st.error(f"**Derivar a evaluación cardiovascular.** El riesgo supera el umbral de {pct(UMBRAL)}.")
            else:
                st.success(f"**Sin indicación de derivación.** El riesgo está bajo el umbral de {pct(UMBRAL)}.")

            # Barra de riesgo vs umbral
            tope = max(0.5, riesgo * 1.15)
            barra = pd.DataFrame({"x": [0], "x2": [riesgo]})
            g = (alt.Chart(pd.DataFrame({"x": [0], "x2": [tope]})).mark_bar(color="#e2e8f0", height=26)
                 .encode(x=alt.X("x:Q", scale=alt.Scale(domain=[0, tope]), axis=alt.Axis(format="%", title=None)), x2="x2")
                 + alt.Chart(barra).mark_bar(color=ROJO if derivar else AZUL, height=26).encode(x="x:Q", x2="x2")
                 + alt.Chart(pd.DataFrame({"u": [UMBRAL]})).mark_rule(color="black", size=3, y=-6, y2=34).encode(x="u:Q"))
            st.altair_chart(g.properties(height=70), use_container_width=True)
            st.caption("La línea negra marca el umbral de derivación.")

            # Referencia: riesgo observado en personas parecidas de la cohorte
            par = cohorte[(cohorte["SEX"] == datos["SEX"]) & cohorte["AGE"].between(edad - 5, edad + 5)]
            if len(par) >= 30:
                st.caption(f"En la cohorte, {pct(par[C.OBJETIVO].mean())} de las personas del mismo sexo y "
                           f"{edad - 5}–{edad + 5} años tuvo un evento coronario en 10 años (n = {len(par)}).")

            # Qué factores empujan el riesgo (contribución al logit respecto de la persona promedio)
            pre = paquete["pipeline"].named_steps["pre"]
            lr = paquete["pipeline"].named_steps["modelo"]
            x = pre.transform(pd.DataFrame([datos])[paquete["variables"]])[0]
            x_ref = pre.transform(cohorte[paquete["variables"]]).mean(axis=0)
            aporte = pd.DataFrame({"Factor": [C.ETIQUETAS[v] for v in paquete["variables"]],
                                   "Aporte": lr.coef_[0] * (x - x_ref)})
            aporte = aporte[aporte["Aporte"].abs() > 0.05].sort_values("Aporte", ascending=False)
            if len(aporte):
                st.markdown("**Qué lo sube y qué lo baja** (frente a la persona promedio de la cohorte)")
                ch = (alt.Chart(aporte).mark_bar().encode(
                    x=alt.X("Aporte:Q", title="Efecto sobre el riesgo (escala logit)"),
                    y=alt.Y("Factor:N", sort="-x", title=None),
                    color=alt.condition("datum.Aporte > 0", alt.value(ROJO), alt.value(AZUL)),
                    tooltip=["Factor", alt.Tooltip("Aporte:Q", format=".2f")]))
                st.altair_chart(ch.properties(height=28 * len(aporte) + 30), use_container_width=True)

# --------------------------------------------------------------------------- #
# 2. Dashboard de resultados
# --------------------------------------------------------------------------- #
with tab_dash:
    m = reporte["prueba"]
    cm = reporte["confusion"]
    st.subheader("Desempeño en el conjunto de prueba")
    st.caption(f"{reporte['datos']['prueba']} personas que el modelo no vio al entrenar · "
               f"{cm['VP'] + cm['FN']} con evento coronario · umbral {pct(UMBRAL)} · entrenado {reporte['fecha'][:10]}")

    k = st.columns(5)
    k[0].metric("ROC-AUC", num(m["ROC-AUC"], 2))
    k[1].metric("Recall", num(m["Recall"], 2), help="De los que tienen un evento, cuántos detecta")
    k[2].metric("Precisión", num(m["Precisión"], 2), help="De los derivados, cuántos tienen un evento")
    k[3].metric("F1", num(m["F1"], 2))
    k[4].metric("Brier", num(m["Brier"]), help="Error de la probabilidad; más bajo es mejor")
    st.caption(f"Accuracy {num(m['Accuracy'], 2)}: con 9 % de casos no sirve para comparar "
               "(decir que nadie se enferma ya acierta 91 %).")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("**Matriz de confusión**")
        mc = pd.DataFrame([
            {"Realidad": "Sin evento", "Predicción": "Sin riesgo", "n": cm["VN"], "Tipo": "Verdaderos negativos"},
            {"Realidad": "Sin evento", "Predicción": "Derivar", "n": cm["FP"], "Tipo": "Falsos positivos"},
            {"Realidad": "Con evento", "Predicción": "Sin riesgo", "n": cm["FN"], "Tipo": "Falsos negativos"},
            {"Realidad": "Con evento", "Predicción": "Derivar", "n": cm["VP"], "Tipo": "Verdaderos positivos"}])
        base = alt.Chart(mc).encode(x=alt.X("Predicción:N", sort=["Sin riesgo", "Derivar"], axis=alt.Axis(labelAngle=0)),
                                    y=alt.Y("Realidad:N", sort=["Sin evento", "Con evento"]))
        heat = base.mark_rect().encode(color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), legend=None),
                                       tooltip=["Tipo", "n"])
        txt = base.mark_text(fontSize=18, fontWeight="bold").encode(
            text="n:Q", color=alt.condition("datum.n > 300", alt.value("white"), alt.value("black")))
        st.altair_chart((heat + txt).properties(height=260), use_container_width=True)
        st.caption(f"Detecta {cm['VP']} de {cm['VP'] + cm['FN']} eventos. Deriva a "
                   f"{pct((cm['VP'] + cm['FP']) / reporte['datos']['prueba'], 0)} de las personas.")

    with c2:
        st.markdown("**Curva ROC**")
        roc = pd.DataFrame(reporte["roc"])
        ch = (alt.Chart(roc).mark_line(color=AZUL, strokeWidth=2.5).encode(
                x=alt.X("fpr:Q", title="1 − especificidad"), y=alt.Y("tpr:Q", title="Sensibilidad (recall)"))
              + alt.Chart(pd.DataFrame({"a": [0, 1]})).mark_line(color=GRIS, strokeDash=[4, 4]).encode(x="a:Q", y="a:Q"))
        st.altair_chart(ch.properties(height=260), use_container_width=True)
        st.caption(f"AUC {num(m['ROC-AUC'], 2)}: ordena correctamente a un par enfermo/sano 3 de cada 4 veces.")

    with c3:
        st.markdown("**Calibración**")
        cal = pd.DataFrame(reporte["calibracion"])
        lim = float(max(cal.max().max(), 0.1)) * 1.1
        ch = (alt.Chart(cal).mark_line(point=True, color=AZUL, strokeWidth=2.5).encode(
                x=alt.X("estimado:Q", title="Riesgo estimado", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, lim])),
                y=alt.Y("observado:Q", title="Proporción observada", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, lim])))
              + alt.Chart(pd.DataFrame({"a": [0, lim]})).mark_line(color=GRIS, strokeDash=[4, 4]).encode(x="a:Q", y="a:Q"))
        st.altair_chart(ch.properties(height=260), use_container_width=True)
        st.caption("Cerca de la diagonal: un 20 % estimado se parece a un 20 % real.")

    st.divider()
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Desempeño por sexo** (mismo umbral)")
        ps = pd.DataFrame(reporte["por_sexo"])
        st.dataframe(ps.style.format({"Recall": "{:.2f}", "Precisión": "{:.2f}", "% derivado": "{:.0%}",
                                      "ROC-AUC": "{:.2f}"}), hide_index=True, use_container_width=True)
        st.caption("Con un umbral único se escapan más casos en mujeres. Se propone evaluar un umbral por sexo; "
                   "con 36 y 40 casos en prueba, las cifras son poco precisas.")
    with c2:
        st.markdown("**Qué factores pesan en el modelo**")
        od = pd.DataFrame(reporte["odds_ratios"])
        od["base"] = 1.0
        ch = (alt.Chart(od).mark_bar().encode(
                x=alt.X("Odds ratio:Q", scale=alt.Scale(type="log", domain=[0.35, 2]), title="Odds ratio (escala log)"),
                x2="base:Q",
                y=alt.Y("Factor:N", sort="-x", title=None),
                color=alt.condition("datum['Odds ratio'] > 1", alt.value(ROJO), alt.value(AZUL)),
                tooltip=["Factor", "Odds ratio", "Por cada"])
              + alt.Chart(pd.DataFrame({"u": [1]})).mark_rule(color="black").encode(x="u:Q"))
        st.altair_chart(ch.properties(height=330), use_container_width=True)
        st.caption("Sobre 1 sube el riesgo, bajo 1 lo baja. Continuas: por cada desviación estándar adicional.")

    exp = C.DIR_RESULTADOS / "experimentos.csv"
    if exp.exists():
        with st.expander("Registro de experimentos de la Parte 2"):
            st.dataframe(pd.read_csv(exp), hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------- #
# 3. Contraste con la ENS 2016-2017
# --------------------------------------------------------------------------- #
with tab_ens:
    ens = reporte.get("ens")
    if not ens:
        st.info("Coloque datos/ens_homologada.csv y vuelva a correr el pipeline para ver este contraste.")
    else:
        st.subheader("¿Cuánto riesgo asigna el modelo a la población chilena?")
        st.caption("ENS 2016-2017, 32 a 70 años sin enfermedad coronaria previa, ponderada por factor de expansión. "
                   "La ENS no tiene seguimiento: esto no mide aciertos, muestra cuánto habría que recalibrar.")
        k = st.columns(4)
        k[0].metric("Riesgo medio estimado (Chile)", pct(ens["riesgo_medio_ponderado"]))
        k[1].metric("Eventos observados en Framingham", pct(ens["prevalencia_framingham"]))
        k[2].metric("Se derivaría en Chile", pct(ens["derivados_ponderado"], 0))
        k[3].metric("Personas ENS", f"{ens['personas']:,}".replace(",", "."))

        er = pd.read_csv(C.DIR_RESULTADOS / "ens_riesgo.csv")
        er["Sexo"] = er["SEX"].map({1: "Hombres", 2: "Mujeres"})
        er["Tramo"] = pd.cut(er["AGE"], [31, 44, 54, 64, 70], labels=["32–44", "45–54", "55–64", "65–70"])
        resumen = (er.groupby(["Tramo", "Sexo"], observed=True)
                     .apply(lambda g: np.average(g["riesgo"], weights=g["PESO"]), include_groups=False)
                     .rename("Riesgo").reset_index())
        ch = alt.Chart(resumen).mark_bar().encode(
            x=alt.X("Tramo:N", title="Edad", axis=alt.Axis(labelAngle=0)), xOffset="Sexo:N",
            y=alt.Y("Riesgo:Q", axis=alt.Axis(format="%"), title="Riesgo medio estimado a 10 años"),
            color=alt.Color("Sexo:N", scale=alt.Scale(range=[AZUL, "#f59e0b"])),
            tooltip=["Tramo", "Sexo", alt.Tooltip("Riesgo:Q", format=".1%")])
        st.altair_chart(ch.properties(height=320), use_container_width=True)
        st.caption(f"Hombres {pct(ens['riesgo_medio_hombres'])} · Mujeres {pct(ens['riesgo_medio_mujeres'])}. "
                   "El siguiente paso es recalibrar el intercepto con tasas de eventos chilenas (DEIS/MINSAL) "
                   "antes de usar el modelo en un CESFAM.")
