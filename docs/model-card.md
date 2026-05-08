# Model Card: Heart Disease Best Model

## Nombre Del Modelo

`heart_disease_best_model`

## Versión Actual

`v3.0.0`

## Algoritmo Seleccionado

El artefacto activo es:

```text
StandardScaler + LogisticRegression
```

Esta versión no se eligió manualmente. `backend/train_model.py` entrena y compara varios modelos, evalúa sus métricas y guarda como artefacto activo el mejor según el criterio definido.

## Modelos Comparados

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC | Estado |
|---|---:|---:|---:|---:|---:|---|
| `StandardScaler + LogisticRegression` | 0.8519 | 0.7857 | 0.9167 | 0.8462 | 0.8958 | Seleccionado como v3. |
| `RandomForestClassifier` | 0.8333 | 0.8000 | 0.8333 | 0.8163 | 0.8806 | Comparado. |
| `GradientBoostingClassifier` | 0.8148 | 0.7500 | 0.8750 | 0.8077 | 0.8861 | Comparado. |
| `StandardScaler + MLPClassifier` | 0.7407 | 0.7083 | 0.7083 | 0.7083 | 0.8847 | Conservado como v2. |
| `StandardScaler + SVC` | 0.8148 | 0.7692 | 0.8333 | 0.8000 | 0.8806 | Comparado. |

El detalle completo está en `models/evaluation_report.json`.

## Criterio De Selección

La métrica principal es `recall`. En una demostración académica de riesgo clínico interesa reducir falsos negativos, porque un falso negativo significa clasificar como bajo/no riesgo un caso que realmente pertenece a la clase `Disease`.

Para evitar seleccionar un modelo con `recall` alto pero comportamiento global débil, el script desempata por:

1. `f1_score`
2. `roc_auc`
3. `accuracy`

El criterio exacto queda registrado en:

```text
models/evaluation_report.json
```

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

## Métricas De Evaluación Del Modelo Activo

| Métrica | Valor |
|---|---:|
| Accuracy | 0.8519 |
| Precision | 0.7857 |
| Recall | 0.9167 |
| F1-score | 0.8462 |
| ROC-AUC | 0.8958 |

Matriz de confusión del modelo seleccionado:

```json
[
  [24, 6],
  [2, 22]
]
```

La imagen generada está en:

```text
docs/images/confusion_matrix.png
```

## Uso Previsto

Este modelo está diseñado para una demostración académica de MLOps:

- Servir un modelo con FastAPI.
- Consumir predicciones desde un frontend React.
- Versionar artefactos de modelo.
- Comparar algoritmos antes de publicar una versión activa.
- Exponer métricas para Prometheus y Grafana.
- Explicar un flujo de despliegue reproducible con Docker Compose.

## Limitaciones

- No debe utilizarse para diagnóstico médico real.
- El dataset es pequeño para estándares clínicos actuales.
- No se ha realizado validación clínica externa ni validación por cohortes independientes.
- Las métricas proceden de un split de test reproducible, no de un estudio clínico.
- `recall` ayuda a reducir falsos negativos, pero puede aumentar falsos positivos.
- La probabilidad devuelta por el modelo debe interpretarse como salida estadística del clasificador, no como probabilidad clínica certificada.
- Las variables proceden del dataset original y pueden no representar todos los factores relevantes en una valoración cardiovascular real.
- Puede haber sesgos por distribución histórica, procedencia y tamaño del dataset.

## Advertencia De Uso Académico

Este proyecto es exclusivamente educativo. El resultado es orientativo y no sustituye una valoración médica profesional, pruebas diagnósticas ni protocolos clínicos.

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

- `models/heart_model_v3_best.joblib`
- `models/model_metadata.json`
- `models/evaluation_report.json`
- `docs/images/confusion_matrix.png`

## Cómo Versionar Una Nueva Versión

1. Entrenar los modelos candidatos con `backend/train_model.py`.
2. Revisar `models/evaluation_report.json`.
3. Guardar el nuevo artefacto con nombre versionado, por ejemplo `models/heart_model_v4_best.joblib`.
4. Actualizar `models/model_metadata.json` con versión, algoritmo, métricas, `selected_metric`, `model_path`, fuente del dataset y fecha.
5. Ejecutar tests, demo local y validación Docker.
6. Documentar el cambio en README y en esta Model Card.

## Riesgos O Sesgos Posibles

- Sesgo por distribución histórica del dataset.
- Representación limitada de grupos demográficos.
- Variables clínicas simplificadas y codificadas.
- Posible sensibilidad a outliers o valores fuera del rango habitual.
- Riesgo de interpretar la salida como diagnóstico en lugar de resultado de demo.
- Riesgo de optimizar una métrica académica sin validación clínica externa.

## Mejoras Futuras Del Modelo

- Añadir validación cruzada estratificada.
- Registrar experimentos con MLflow.
- Calibrar probabilidades.
- Añadir explicabilidad con SHAP u otra técnica interpretable.
- Incorporar un dataset curado local con trazabilidad clara.
- Automatizar el versionado de modelos en CI/CD.
