"""Parámetros del proyecto Caso 5 (riesgo coronario a 10 años).

Todo lo que define el modelo está aquí, para que el pipeline y la app
usen exactamente las mismas variables, rutas y semilla que la Parte 2.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "datos"
DIR_RESULTADOS = RAIZ / "resultados"

ARCHIVO_COHORTE = "framingham_cohorte.csv"   # salida de la Parte 1
ARCHIVO_ENS = "ens_homologada.csv"           # población chilena para contraste
ARCHIVO_MODELO = "modelo_chd10a.pkl"

# Variables del modelo (mismo orden que la Parte 2)
CONT = ["AGE", "SYSBP", "DIABP", "TOTCHOL", "GLUCOSE", "BMI", "HEARTRTE", "CIGPDAY"]
BIN = ["SEX", "CURSMOKE", "DIABETES", "BPMEDS"]
PRED = CONT + BIN
OBJETIVO = "chd_10a"

# Decisiones de la Parte 2
SEMILLA = 42
TEST_SIZE = 0.20
N_PLIEGUES = 5
C_FINAL = 10            # mejor C de la grilla v2
RECALL_OBJETIVO = 0.75  # el umbral se elige para detectar al menos 3 de cada 4 casos

# Rangos válidos para el formulario y la validación de entradas
# (min, max, valor por defecto, etiqueta, unidad)
RANGOS = {
    "AGE":      (30, 75, 55, "Edad", "años"),
    "SYSBP":    (80, 250, 130, "Presión sistólica", "mmHg"),
    "DIABP":    (40, 150, 80, "Presión diastólica", "mmHg"),
    "TOTCHOL":  (100, 450, 220, "Colesterol total", "mg/dL"),
    "GLUCOSE":  (40, 400, 90, "Glicemia", "mg/dL"),
    "BMI":      (15.0, 60.0, 26.0, "IMC", "kg/m²"),
    "HEARTRTE": (40, 140, 75, "Frecuencia cardíaca", "lpm"),
    "CIGPDAY":  (0, 80, 0, "Cigarrillos al día", "cig/día"),
}

ETIQUETAS = {
    **{k: v[3] for k, v in RANGOS.items()},
    "SEX": "Sexo",
    "CURSMOKE": "Fumador actual",
    "DIABETES": "Diabetes",
    "BPMEDS": "Usa antihipertensivos",
}
