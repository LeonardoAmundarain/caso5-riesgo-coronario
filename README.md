# Caso 5 · Riesgo coronario a 10 años

**Gestión de Proyectos de Datos (SCY1102) · Duoc UC · Parte 3: Entregable funcional**

Modelo de tamizaje que estima el riesgo de un evento coronario en los próximos 10 años a partir de
datos de un control preventivo (edad, presión, colesterol, glicemia, IMC, tabaquismo, diabetes y uso
de antihipertensivos), e indica si conviene derivar a evaluación cardiovascular.

| | |
|---|---|
| Datos de entrenamiento | Cohorte de Framingham, 4.224 personas, 9 % con evento (Parte 1) |
| Modelo | Regresión logística sin pesos de clase (C = 10) + umbral de derivación (Parte 2) |
| Umbral | 7,7 % de riesgo: el más alto que detecta al menos 75 % de los casos en validación cruzada |
| Prueba (845 personas) | ROC-AUC 0,76 · Recall 0,74 · Precisión 0,17 · Brier 0,076 |
| Contraste | ENS 2016-2017 ponderada: riesgo medio estimado 6,6 % en Chile |

> Prototipo académico. 

## Estructura

```
caso5-riesgo-coronario/
├── datos/
│   ├── heart.csv                # dataset original del caso (se descarta en la Parte 1)
│   ├── framingham_teaching.csv  # Framingham docente, insumo de la Parte 1
│   ├── framingham_cohorte.csv   # salida de la Parte 1, entrada del modelo
│   └── ens_homologada.csv       # ENS 2016-2017 con las mismas variables
├── notebooks/
│   ├── Parte1a_Limpieza_heart_Caso5.ipynb              # EDA y diagnóstico de heart.csv
│   ├── Parte1b_Preparacion_Framingham_ENS_Caso5.ipynb  # cohorte Framingham + ENS
│   └── Parte2_Modelo_Caso5.ipynb                       # comparación de modelos (v1–v3)
├── src/
│   ├── config.py                # variables, rutas, semilla y rangos válidos
│   └── pipeline.py              # cargar → procesar → entrenar → evaluar → guardar
├── resultados/                  # lo genera el pipeline
│   ├── modelo_chd10a.pkl        # pipeline entrenado + umbral + variables
│   ├── reporte_modelo.json      # métricas, matriz, ROC, calibración, por sexo, odds, ENS
│   ├── predicciones_prueba.csv
│   ├── ens_riesgo.csv
│   └── experimentos.csv         # registro de la Parte 2
├── tests/test_pipeline.py       # 8 pruebas, incluida la reproducción de la Parte 2
├── app.py                       # prototipo en Streamlit
├── docs/capturas/               # pantallas del prototipo
└── requirements.txt
```

## Etapas del proyecto

| Parte | Qué se hizo | Dónde |
|---|---|---|
| 1. Datos | EDA de heart.csv y decisión de descartarlo; preparación de Framingham y ENS | `notebooks/Parte1a…`, `notebooks/Parte1b…` |
| 2. Modelo | Regresión logística, random forest y XGBoost; métricas y registro de experimentos | `notebooks/Parte2…`, `resultados/experimentos.csv` |
| 3. Entregable | Pipeline reproducible, prototipo en Streamlit, pruebas y este repositorio | `src/`, `app.py`, `tests/` |

Los notebooks corren en Google Colab montando la carpeta del proyecto en Drive, o en local desde la raíz
del repositorio (leen `datos/` y escriben en `resultados/`).

## Cómo ejecutarlo

```bash
pip install -r requirements.txt

# 1. Entrenar y guardar el modelo (≈ 2 segundos)
python -m src.pipeline

# 2. Comprobar que todo funciona
python -m pytest -q

# 3. Abrir el prototipo
streamlit run app.py
```

Opciones del pipeline:

```bash
python -m src.pipeline --datos otra/carpeta --salida otra/salida   # otras rutas
python -m src.pipeline --sin-ens                                   # sin el contraste con Chile
```

### En Google Colab

```python
from google.colab import drive; drive.mount("/content/drive")
%cd "/content/drive/MyDrive/Evalucion 3 y Examen/caso5-riesgo-coronario"
!pip install -q -r requirements.txt
!python -m src.pipeline
```

## Qué hace el pipeline

1. **Carga y valida** la cohorte: columnas requeridas, sin IDs repetidos y etiqueta 0/1.
2. **Divide** 80/20 estratificado (`random_state=42`), igual que la Parte 2.
3. **Procesa** dentro de un `Pipeline` de scikit-learn: mediana + escalado en continuas, moda en
   binarias. Así las medianas y el escalado se calculan solo con el entrenamiento.
4. **Elige el umbral** con predicciones de validación cruzada (5 pliegues) sobre el entrenamiento.
5. **Entrena** con todo el entrenamiento y **evalúa una sola vez** en la prueba.
6. **Guarda** el modelo, un reporte JSON con todo lo que muestra el dashboard y las predicciones.
7. **Contrasta** con la ENS ponderada para estimar cuánto habría que recalibrar.

Con la misma semilla, cada ejecución entrega exactamente las mismas métricas; el test
`test_reproduce_parte2` lo comprueba contra los números del cuaderno de la Parte 2.

## El prototipo

- **Evaluar a una persona:** formulario con validación de rangos (p. ej., la diastólica debe ser menor
  que la sistólica). Entrega el riesgo, la indicación de derivar o no, el riesgo observado en personas
  parecidas de la cohorte y qué factores lo suben o lo bajan.
- **Resultados del modelo:** métricas de prueba, matriz de confusión, curva ROC, calibración,
  desempeño por sexo, odds ratios y registro de experimentos.
- **Contraste con Chile:** riesgo medio estimado en la ENS por sexo y tramo de edad.

## Usar el modelo desde otro código

```python
from src.pipeline import cargar_modelo, predecir
m = cargar_modelo()
riesgo, derivar = predecir(m, {"AGE": 60, "SEX": 1, "SYSBP": 150, "DIABP": 90, "TOTCHOL": 260,
                               "GLUCOSE": 100, "BMI": 28, "HEARTRTE": 75, "CURSMOKE": 1,
                               "CIGPDAY": 20, "DIABETES": 0, "BPMEDS": 0})
# riesgo ≈ 0,301 → derivar = True
```

## Limitaciones conocidas

- Con un umbral único, el modelo detecta 86 % de los casos en hombres y 62 % en mujeres. Se propone
  evaluar un umbral por sexo.
- Está entrenado con población estadounidense de mediados del siglo XX. Antes de usarlo en Chile hay
  que recalibrar el intercepto con tasas de eventos locales.



## Equipo

Leonardo Amundarain · Felipe Villalobos · Solange Valdebenito · Adolfo 
