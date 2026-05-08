# Model Card: Heart Disease MLP

## Nombre Del Modelo

`heart_disease_mlp`

## Versión Actual

`v2.0.0`

## Algoritmo

Pipeline de scikit-learn compuesto por:

```text
StandardScaler + MLPClassifier
```

El escalado estandariza las variables numéricas antes de pasarlas a la red neuronal multicapa. `MLPClassifier` se usa como una red neuronal sencilla, reproducible y fácil de serializar con `joblib`.

## Dataset Utilizado

El ZIP del taller base no incluye un CSV local. El script de entrenamiento está preparado para usar:

1. `data/heart.csv`, si se añade manualmente.
2. Un CSV indicado con `--data-path`.
3. El dataset `heart-statlog` de OpenML, que es la fuente utilizada por el notebook original del taller.

No se genera dataset sintético en esta versión del proyecto.

## Variables De Entrada

| Variable | Descripción |
|---|---|
| `age` | Edad del paciente. |
| `sex` | Sexo codificado. |
| `chest` | Tipo de dolor torácico. |
| `resting_blood_pressure` | Presión arterial en reposo. |
| `serum_cholestoral` | Colesterol sérico. |
| `fasting_blood_sugar` | Glucosa en ayunas. |
| `resting_electrocardiographic_results` | Resultado electrocardiográfico en reposo. |
| `maximum_heart_rate_achieved` | Frecuencia cardíaca máxima alcanzada. |
| `exercise_induced_angina` | Angina inducida por ejercicio. |
| `oldpeak` | Depresión ST inducida por ejercicio. |
| `slope` | Pendiente del segmento ST. |
| `number_of_major_vessels` | Número de vasos principales observados. |
| `thal` | Variable thal del dataset original. |

## Variable Objetivo

Predicción binaria de presencia de enfermedad cardíaca:

- `0`: No Disease
- `1`: Disease

## Métricas De Evaluación

Las métricas principales se registran en `models/model_metadata.json` y en `models/evaluation_report.json`.

La versión actual registra:

| Métrica | Valor |
|---|---:|
| Accuracy | 0.7407 |
| Precision | 0.7083 |
| Recall | 0.7083 |
| F1-score | 0.7083 |

El reporte de evaluación también incluye la matriz de confusión:

```json
[
  [23, 7],
  [7, 17]
]
```

Esta evaluación procede del split de test utilizado por `backend/train_model.py` sobre la fuente `OpenML heart-statlog`.

## Uso Previsto

Este modelo está diseñado para una demostración académica de MLOps:

- Servir un modelo con FastAPI.
- Consumir predicciones desde un frontend React.
- Versionar artefactos de modelo.
- Exponer métricas para Prometheus y Grafana.
- Explicar un flujo de despliegue reproducible con Docker Compose.

## Limitaciones

- No debe utilizarse para diagnóstico médico real.
- El rendimiento depende del dataset disponible y de su calidad.
- El dataset usado es pequeño para estándares clínicos actuales.
- No se ha realizado validación clínica externa.
- La métrica `accuracy` no refleja por sí sola todos los riesgos de falsos positivos o falsos negativos.
- `precision`, `recall` y `f1_score` ayudan a revisar el comportamiento del clasificador, pero no sustituyen una validación clínica.
- La probabilidad devuelta por el modelo debe interpretarse como salida del clasificador, no como probabilidad clínica certificada.

## Advertencia De Uso Académico

Este proyecto es exclusivamente educativo. No sustituye el criterio de profesionales sanitarios, pruebas diagnósticas ni protocolos clínicos.

## Cómo Reentrenarlo

Con la fuente OpenML por defecto:

```bash
python backend/train_model.py
```

Con un CSV local:

```bash
python backend/train_model.py --data-path data/heart.csv
```

El entrenamiento genera:

- `models/heart_model_v2_mlp.joblib`
- `models/model_metadata.json`
- `models/evaluation_report.json`

## Cómo Versionar Una Nueva Versión

1. Entrenar el nuevo modelo.
2. Guardarlo con un nombre versionado, por ejemplo `models/heart_model_v3_mlp.joblib`.
3. Generar `models/evaluation_report.json` con accuracy, precision, recall, F1, matriz de confusión, fuente del dataset y fecha.
4. Actualizar `models/model_metadata.json` con la nueva versión, métricas, fecha, `model_path` y `evaluation_report_path`.
5. Ejecutar tests y una demo local.
6. Documentar el cambio en README y en esta Model Card.

## Riesgos O Sesgos Posibles

- Sesgo por distribución histórica del dataset.
- Representación limitada de grupos demográficos.
- Variables clínicas simplificadas y codificadas.
- Posible sensibilidad a outliers o valores fuera del rango habitual.
- Riesgo de interpretar la salida como diagnóstico en lugar de resultado de demo.

## Mejoras Futuras Del Modelo

- Añadir validación cruzada y comparación de varios algoritmos.
- Registrar experimentos con MLflow.
- Medir recall, precisión, F1, ROC-AUC y matriz de confusión.
- Calibrar probabilidades.
- Añadir explicabilidad con SHAP u otra técnica interpretable.
- Incorporar un dataset curado local con trazabilidad clara.
- Automatizar el versionado de modelos en CI/CD.
