"""Pipeline reproducible del Caso 5: riesgo coronario a 10 años.

Toma la cohorte de Framingham que dejó la Parte 1, la procesa, entrena el
modelo elegido en la Parte 2 (regresión logística sin pesos + umbral de
derivación), lo evalúa en el conjunto de prueba y guarda todo en resultados/.

Uso:
    python -m src.pipeline                      # rutas por defecto
    python -m src.pipeline --datos ruta/datos --salida ruta/resultados
    python -m src.pipeline --sin-ens            # omite el contraste con la ENS

Con la misma semilla, el resultado es idéntico en cada ejecución.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C

log = logging.getLogger("caso5")


# --------------------------------------------------------------------------- #
# 1. Carga y validación
# --------------------------------------------------------------------------- #
def cargar_cohorte(ruta: Path) -> pd.DataFrame:
    """Lee la cohorte y verifica que tenga lo que el modelo necesita."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró la cohorte en {ruta}")
    df = pd.read_csv(ruta)

    faltan = [c for c in C.PRED + [C.OBJETIVO] if c not in df.columns]
    if faltan:
        raise ValueError(f"A la cohorte le faltan columnas: {faltan}")
    if "ID" in df.columns and df["ID"].duplicated().any():
        raise ValueError("Hay personas repetidas (ID duplicado) en la cohorte")
    if not set(df[C.OBJETIVO].unique()) <= {0, 1}:
        raise ValueError(f"{C.OBJETIVO} debe ser 0 o 1")

    log.info("Cohorte: %d personas, %d casos (%.1f %%)",
             len(df), df[C.OBJETIVO].sum(), 100 * df[C.OBJETIVO].mean())
    return df


def dividir(df: pd.DataFrame):
    """Partición 80/20 estratificada, igual a la Parte 2."""
    X, y = df[C.PRED], df[C.OBJETIVO]
    return train_test_split(X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.SEMILLA)


# --------------------------------------------------------------------------- #
# 2. Procesamiento + modelo
# --------------------------------------------------------------------------- #
def preprocesador() -> ColumnTransformer:
    """Mediana + escalado en continuas, moda en binarias (definido en la Parte 1)."""
    return ColumnTransformer([
        ("cont", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("esc", StandardScaler())]), C.CONT),
        ("bin", SimpleImputer(strategy="most_frequent"), C.BIN),
    ])


def construir_modelo(c: float = C.C_FINAL) -> Pipeline:
    """Regresión logística sin pesos de clase, para que las probabilidades sean realistas."""
    return Pipeline([("pre", preprocesador()),
                     ("modelo", LogisticRegression(C=c, max_iter=1000))])


def elegir_umbral(modelo: Pipeline, X_train, y_train, recall_objetivo: float = C.RECALL_OBJETIVO):
    """Umbral más alto que mantiene el recall objetivo en validación cruzada.

    Se calcula solo con entrenamiento; el conjunto de prueba no participa.
    """
    cv = StratifiedKFold(n_splits=C.N_PLIEGUES, shuffle=True, random_state=C.SEMILLA)
    p_oof = cross_val_predict(modelo, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    prec, rec, umbrales = precision_recall_curve(y_train, p_oof)
    umbral = float(umbrales[rec[:-1] >= recall_objetivo].max())
    cv_info = {
        "roc_auc": round(float(roc_auc_score(y_train, p_oof)), 3),
        "recall": round(float(recall_score(y_train, p_oof >= umbral)), 3),
        "precision": round(float(precision_score(y_train, p_oof >= umbral)), 3),
        "brier": round(float(brier_score_loss(y_train, p_oof)), 3),
        "derivados": round(float((p_oof >= umbral).mean()), 3),
    }
    log.info("Umbral elegido: %.3f (recall en CV %.3f)", umbral, cv_info["recall"])
    return umbral, cv_info


# --------------------------------------------------------------------------- #
# 3. Evaluación
# --------------------------------------------------------------------------- #
def evaluar(modelo: Pipeline, umbral: float, X_test, y_test) -> dict:
    p = modelo.predict_proba(X_test)[:, 1]
    pred = (p >= umbral).astype(int)
    vn, fp, fn, vp = confusion_matrix(y_test, pred).ravel()
    fpr, tpr, _ = roc_curve(y_test, p)
    obs, esp = calibration_curve(y_test, p, n_bins=8, strategy="quantile")

    metricas = {
        "ROC-AUC": roc_auc_score(y_test, p),
        "PR-AUC": average_precision_score(y_test, p),
        "Recall": recall_score(y_test, pred),
        "Precisión": precision_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "Accuracy": accuracy_score(y_test, pred),
        "Brier": brier_score_loss(y_test, p),
    }
    return {
        "metricas": {k: round(float(v), 3) for k, v in metricas.items()},
        "confusion": {"VN": int(vn), "FP": int(fp), "FN": int(fn), "VP": int(vp)},
        "roc": {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()},
        "calibracion": {"estimado": esp.round(4).tolist(), "observado": obs.round(4).tolist()},
        "p": p, "pred": pred,
    }


def por_sexo(X_test, y_test, p, pred) -> pd.DataFrame:
    res = pd.DataFrame({"Sexo": X_test["SEX"].map({1: "Hombres", 2: "Mujeres"}).values,
                        "y": y_test.values, "p": p, "pred": pred})
    filas = []
    for sexo, g in res.groupby("Sexo"):
        filas.append({"Sexo": sexo, "Personas": len(g), "Casos": int(g.y.sum()),
                      "Recall": recall_score(g.y, g.pred),
                      "Precisión": precision_score(g.y, g.pred),
                      "% derivado": g.pred.mean(),
                      "ROC-AUC": roc_auc_score(g.y, g.p)})
    return pd.DataFrame(filas).round(3)


def odds_ratios(modelo: Pipeline) -> pd.DataFrame:
    lr = modelo.named_steps["modelo"]
    por_cada = ["+1 desviación estándar"] * len(C.CONT) + \
               ["mujer vs. hombre", "fumador", "diabetes", "usa antihipertensivos"]
    return (pd.DataFrame({"Variable": C.PRED,
                          "Factor": [C.ETIQUETAS[v] for v in C.PRED],
                          "Odds ratio": np.exp(lr.coef_[0]).round(2),
                          "Por cada": por_cada})
            .sort_values("Odds ratio", ascending=False).reset_index(drop=True))


def contraste_ens(modelo: Pipeline, umbral: float, ruta: Path, prevalencia_framingham: float):
    """Aplica el modelo a la ENS 2016-2017 (ponderada) para ver cuánto riesgo asigna en Chile.

    La ENS no tiene seguimiento, así que esto no mide aciertos: muestra cuánto
    se desplaza el riesgo estimado y sirve para planificar la recalibración.
    """
    ens = pd.read_csv(ruta)
    ens = ens[(ens["PREVCHD"] != 1) & ens["AGE"].between(32, 70)].copy()  # mismo criterio que la cohorte
    ens["riesgo"] = modelo.predict_proba(ens[C.PRED])[:, 1]
    w = ens["PESO"]
    resumen = {
        "personas": int(len(ens)),
        "riesgo_medio_ponderado": round(float(np.average(ens["riesgo"], weights=w)), 4),
        "derivados_ponderado": round(float(np.average(ens["riesgo"] >= umbral, weights=w)), 4),
        "prevalencia_framingham": round(float(prevalencia_framingham), 4),
    }
    for sexo, nombre in [(1, "hombres"), (2, "mujeres")]:
        s = ens[ens["SEX"] == sexo]
        resumen[f"riesgo_medio_{nombre}"] = round(float(np.average(s["riesgo"], weights=s["PESO"])), 4)
    log.info("ENS: riesgo medio ponderado %.1f %%, derivados %.1f %%",
             100 * resumen["riesgo_medio_ponderado"], 100 * resumen["derivados_ponderado"])
    return resumen, ens[["ID", "SEX", "AGE", "riesgo", "PESO"]]


# --------------------------------------------------------------------------- #
# 4. Uso del modelo guardado (lo usa la app)
# --------------------------------------------------------------------------- #
def validar_persona(datos: dict) -> list[str]:
    """Devuelve la lista de problemas en los datos de una persona (vacía si está todo bien)."""
    errores = []
    for v in C.PRED:
        if v not in datos or datos[v] is None:
            errores.append(f"Falta {C.ETIQUETAS[v]}")
    for v, (mn, mx, *_rest) in C.RANGOS.items():
        x = datos.get(v)
        if x is not None and not (mn <= x <= mx):
            errores.append(f"{C.ETIQUETAS[v]} fuera de rango ({mn}–{mx})")
    if datos.get("SEX") not in (1, 2):
        errores.append("Sexo debe ser 1 (hombre) o 2 (mujer)")
    for v in ["CURSMOKE", "DIABETES", "BPMEDS"]:
        if datos.get(v) not in (0, 1):
            errores.append(f"{C.ETIQUETAS[v]} debe ser 0 o 1")
    if datos.get("CURSMOKE") == 0 and (datos.get("CIGPDAY") or 0) > 0:
        errores.append("Si no fuma, los cigarrillos al día deben ser 0")
    if datos.get("SYSBP") is not None and datos.get("DIABP") is not None \
            and datos["DIABP"] >= datos["SYSBP"]:
        errores.append("La presión diastólica debe ser menor que la sistólica")
    return errores


def cargar_modelo(ruta: Path = C.DIR_RESULTADOS / C.ARCHIVO_MODELO) -> dict:
    return joblib.load(ruta)


def predecir(paquete: dict, datos: dict) -> tuple[float, bool]:
    persona = pd.DataFrame([datos])[paquete["variables"]]
    riesgo = float(paquete["pipeline"].predict_proba(persona)[0, 1])
    return riesgo, riesgo >= paquete["umbral"]


# --------------------------------------------------------------------------- #
# 5. Orquestación
# --------------------------------------------------------------------------- #
def ejecutar(dir_datos: Path = C.DIR_DATOS, dir_salida: Path = C.DIR_RESULTADOS,
             con_ens: bool = True) -> dict:
    dir_salida.mkdir(parents=True, exist_ok=True)

    df = cargar_cohorte(dir_datos / C.ARCHIVO_COHORTE)
    X_train, X_test, y_train, y_test = dividir(df)
    log.info("Entrenamiento: %d | Prueba: %d", len(X_train), len(X_test))

    modelo = construir_modelo()
    umbral, cv_info = elegir_umbral(modelo, X_train, y_train)
    modelo.fit(X_train, y_train)

    ev = evaluar(modelo, umbral, X_test, y_test)
    log.info("Prueba: %s", ev["metricas"])
    sexo = por_sexo(X_test, y_test, ev["p"], ev["pred"])
    odds = odds_ratios(modelo)

    resumen_ens = None
    if con_ens and (dir_datos / C.ARCHIVO_ENS).exists():
        resumen_ens, ens_riesgo = contraste_ens(modelo, umbral, dir_datos / C.ARCHIVO_ENS,
                                                prevalencia_framingham=df[C.OBJETIVO].mean())
        ens_riesgo.to_csv(dir_salida / "ens_riesgo.csv", index=False)

    # Modelo: mismo formato que dejó la Parte 2, más metadatos de trazabilidad
    paquete = {
        "pipeline": modelo,
        "umbral": umbral,
        "variables": C.PRED,
        "metricas_prueba": ev["metricas"],
        "sklearn": sklearn.__version__,
        "entrenado": datetime.now().isoformat(timespec="seconds"),
        "semilla": C.SEMILLA,
        "n_entrenamiento": int(len(X_train)),
    }
    joblib.dump(paquete, dir_salida / C.ARCHIVO_MODELO)

    reporte = {
        "fecha": paquete["entrenado"],
        "umbral": round(umbral, 4),
        "validacion_cruzada": cv_info,
        "prueba": ev["metricas"],
        "confusion": ev["confusion"],
        "roc": ev["roc"],
        "calibracion": ev["calibracion"],
        "por_sexo": sexo.to_dict(orient="records"),
        "odds_ratios": odds.to_dict(orient="records"),
        "ens": resumen_ens,
        "datos": {"personas": int(len(df)), "casos": int(df[C.OBJETIVO].sum()),
                  "entrenamiento": int(len(X_train)), "prueba": int(len(X_test))},
    }
    (dir_salida / "reporte_modelo.json").write_text(json.dumps(reporte, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    pd.DataFrame({"riesgo": ev["p"].round(5), "derivar": ev["pred"], "evento": y_test.values,
                  "SEX": X_test["SEX"].values}).to_csv(dir_salida / "predicciones_prueba.csv", index=False)

    log.info("Listo. Archivos en %s", dir_salida)
    return reporte


def main():
    ap = argparse.ArgumentParser(description="Entrena y guarda el modelo de riesgo coronario (Caso 5)")
    ap.add_argument("--datos", type=Path, default=C.DIR_DATOS, help="carpeta con framingham_cohorte.csv")
    ap.add_argument("--salida", type=Path, default=C.DIR_RESULTADOS, help="carpeta donde se guardan los resultados")
    ap.add_argument("--sin-ens", action="store_true", help="no calcular el contraste con la ENS")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    ejecutar(args.datos, args.salida, con_ens=not args.sin_ens)


if __name__ == "__main__":
    main()
