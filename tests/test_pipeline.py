"""Pruebas del pipeline. Ejecutar con:  python -m pytest -q"""
import json

import pandas as pd
import pytest

from src import config as C
from src import pipeline as P


@pytest.fixture(scope="module")
def salida(tmp_path_factory):
    d = tmp_path_factory.mktemp("resultados")
    reporte = P.ejecutar(C.DIR_DATOS, d, con_ens=True)
    return d, reporte


def test_reproduce_parte2(salida):
    """Con la misma semilla debe dar lo mismo que el cuaderno de la Parte 2."""
    _, r = salida
    assert r["datos"] == {"personas": 4224, "casos": 379, "entrenamiento": 3379, "prueba": 845}
    assert r["umbral"] == pytest.approx(0.077, abs=0.001)
    assert r["prueba"]["ROC-AUC"] == pytest.approx(0.762, abs=0.002)
    assert r["prueba"]["Recall"] == pytest.approx(0.737, abs=0.002)
    assert r["confusion"] == {"VN": 502, "FP": 267, "FN": 20, "VP": 56}


def test_es_determinista(salida, tmp_path):
    _, r1 = salida
    r2 = P.ejecutar(C.DIR_DATOS, tmp_path, con_ens=False)
    assert r1["prueba"] == r2["prueba"] and r1["umbral"] == r2["umbral"]


def test_archivos_generados(salida):
    d, _ = salida
    for f in ["modelo_chd10a.pkl", "reporte_modelo.json", "predicciones_prueba.csv", "ens_riesgo.csv"]:
        assert (d / f).exists(), f
    json.loads((d / "reporte_modelo.json").read_text(encoding="utf-8"))


def test_paquete_compatible_con_parte2(salida):
    d, _ = salida
    m = P.cargar_modelo(d / C.ARCHIVO_MODELO)
    assert {"pipeline", "umbral", "variables", "metricas_prueba"} <= set(m)
    assert m["variables"] == C.PRED


def test_prediccion_ejemplo_parte2(salida):
    """Persona de ejemplo del cuaderno de la Parte 2: riesgo 30,1 % → derivar."""
    d, _ = salida
    m = P.cargar_modelo(d / C.ARCHIVO_MODELO)
    persona = {"AGE": 60, "SEX": 1, "SYSBP": 150, "DIABP": 90, "TOTCHOL": 260, "GLUCOSE": 100,
               "BMI": 28, "HEARTRTE": 75, "CURSMOKE": 1, "CIGPDAY": 20, "DIABETES": 0, "BPMEDS": 0}
    riesgo, derivar = P.predecir(m, persona)
    assert riesgo == pytest.approx(0.301, abs=0.002)
    assert derivar


def test_riesgo_sube_con_edad(salida):
    d, _ = salida
    m = P.cargar_modelo(d / C.ARCHIVO_MODELO)
    base = {"SEX": 2, "SYSBP": 120, "DIABP": 78, "TOTCHOL": 200, "GLUCOSE": 85, "BMI": 24,
            "HEARTRTE": 72, "CURSMOKE": 0, "CIGPDAY": 0, "DIABETES": 0, "BPMEDS": 0}
    r40, _ = P.predecir(m, {**base, "AGE": 40})
    r65, _ = P.predecir(m, {**base, "AGE": 65})
    assert r65 > r40


def test_validacion_entradas():
    ok = {"AGE": 50, "SEX": 1, "SYSBP": 130, "DIABP": 80, "TOTCHOL": 220, "GLUCOSE": 90, "BMI": 26,
          "HEARTRTE": 70, "CURSMOKE": 0, "CIGPDAY": 0, "DIABETES": 0, "BPMEDS": 0}
    assert P.validar_persona(ok) == []
    assert P.validar_persona({**ok, "AGE": 120})
    assert P.validar_persona({**ok, "DIABP": 140})            # diastólica >= sistólica
    assert P.validar_persona({**ok, "CIGPDAY": 10})           # no fuma pero declara cigarrillos


def test_cohorte_invalida(tmp_path):
    df = pd.read_csv(C.DIR_DATOS / C.ARCHIVO_COHORTE).drop(columns=["GLUCOSE"])
    df.to_csv(tmp_path / C.ARCHIVO_COHORTE, index=False)
    with pytest.raises(ValueError):
        P.cargar_cohorte(tmp_path / C.ARCHIVO_COHORTE)
