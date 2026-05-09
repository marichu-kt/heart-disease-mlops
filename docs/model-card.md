# Model Card: Heart Disease Robust MLP

## Nombre Del Modelo

`heart_disease_mlp_calibrated`

## Versión Actual

`v4.2.0`

## Algoritmo Activo

```text
StandardScaler + Robust Calibrated MLPClassifier
```

El modelo activo es una red neuronal multicapa (`MLPClassifier`) integrada en un `Pipeline` de scikit-learn con `StandardScaler`. La versión v4.2 mantiene la red neuronal como artefacto final, añade validación cruzada repetida, usa F2-score como métrica principal, ajusta el threshold de decisión y evalúa calibración de probabilidades.

La calibración sigmoid se evaluó con `CalibratedClassifierCV`, pero no se aplicó al artefacto final porque reducía F2-score y no mejoraba suficientemente el Brier score.

## Dataset Utilizado

Fuente principal:

```text
OpenML heart-statlog
```

Origen y justificación:

Se utilizó OpenML `heart-statlog` porque era el dataset original del taller base. El notebook ya lo cargaba mediante `fetch_openml('heart-statlog')`, y las 13 variables clínicas del dataset coincidían con la estructura de entrada esperada por la API. Por eso se mantuvo como dataset principal para conservar compatibilidad, reproducibilidad y coherencia con el ejercicio original.

El dataset contiene 270 registros/pacientes. Es suficiente para un proyecto académico y MLOps demostrativo, pero su tamaño reducido es una limitación relevante. Para un caso real se debería sustituir por un dataset mayor, trazable y compatible, por ejemplo usando `data/heart.csv`.

No se usaron datos sintéticos como dataset principal.

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

## Entrenamiento Y Optimización

La versión v4.2 usa:

- `train_test_split` estratificado con `test_size=0.2` y `random_state=42`.
- `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)`.
- `RandomizedSearchCV` con scoring `F2-score`.
- `MLPClassifier(solver="adam", early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=42)`.
- Ajuste de threshold sobre probabilidades de la clase `Disease`.
- Evaluación de calibración sigmoid con `CalibratedClassifierCV`.

Espacio de búsqueda utilizado:

- `hidden_layer_sizes`: `(16,)`, `(32,)`, `(64,)`, `(32, 16)`, `(64, 32)`, `(128, 64)`
- `activation`: `relu`, `tanh`
- `alpha`: `0.0001`, `0.001`, `0.01`, `0.05`, `0.1`
- `learning_rate_init`: `0.0005`, `0.001`, `0.005`, `0.01`
- `batch_size`: `16`, `32`, `64`
- `learning_rate`: `constant`, `adaptive`
- `max_iter`: `800`, `1000`, `1500`

## Mejor Configuración Encontrada

```json
{
  "hidden_layer_sizes": [64, 32],
  "activation": "tanh",
  "alpha": 0.0001,
  "learning_rate_init": 0.0005,
  "batch_size": 64,
  "learning_rate": "adaptive",
  "max_iter": 1500
}
```

## Criterio De Selección

La búsqueda de hiperparámetros usa F2-score porque en una demostración académica de riesgo clínico interesa dar más peso al recall sin ignorar la precision.

El threshold final se eligió con esta prioridad:

1. mayor `f2_score`;
2. en empate, mayor `f1_score`;
3. en empate, mayor `recall`;
4. en empate, mayor `precision`;
5. en empate, mayor `accuracy`.

## Threshold De Decisión

Threshold activo:

```text
0.35
```

La API lee `decision_threshold` desde `models/model_metadata.json` y lo usa durante inferencia:

```text
probability_disease >= decision_threshold -> Disease
probability_disease < decision_threshold  -> No Disease
```

## Métricas Finales Del Modelo Activo

| Métrica | Valor |
|---|---:|
| Accuracy | 0.7222 |
| Precision | 0.6216 |
| Recall | 0.9583 |
| F1-score | 0.7541 |
| F2-score | 0.8647 |
| ROC-AUC | 0.8806 |
| Average precision | 0.8764 |
| Brier score | 0.1483 |

Matriz de confusión:

```json
[
  [16, 14],
  [1, 23]
]
```

Lectura:

- verdaderos negativos: 16;
- falsos positivos: 14;
- falsos negativos: 1;
- verdaderos positivos: 23.

Imagen principal:

```text
docs/images/confusion_matrix_mlp_v4_2.png
```

## Calibración

| Variante | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | Brier | Decisión |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Sin calibrar | 0.7222 | 0.6216 | 0.9583 | 0.7541 | 0.8647 | 0.8806 | 0.1483 | Activa |
| Sigmoid calibrada | 0.7963 | 0.7241 | 0.8750 | 0.7925 | 0.8400 | 0.8597 | 0.1540 | Evaluada, no aplicada |

La calibración no se activó porque redujo recall, F2-score y ROC-AUC. La decisión queda registrada en `models/evaluation_report.json`.

## Curvas Y Artefactos Generados

| Artefacto | Ruta |
|---|---|
| Matriz de confusión v4.2 | `docs/images/confusion_matrix_mlp_v4_2.png` |
| Curva ROC v4.2 | `docs/images/roc_curve_mlp_v4_2.png` |
| Curva Precision-Recall v4.2 | `docs/images/precision_recall_curve_mlp_v4_2.png` |
| Importancia por permutación v4.2 | `docs/images/feature_importance_mlp_v4_2.png` |
| Métricas por threshold v4.2 | `docs/images/threshold_metrics_mlp_v4_2.png` |
| Comparativa de modelos | `docs/images/model_comparison_metrics.png` |
| Reporte JSON | `models/evaluation_report.json` |

## Comparación Con Versiones Anteriores

| Versión | Modelo | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | Lectura |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v3.0.0 | `StandardScaler + LogisticRegression` | 0.8519 | 0.7857 | 0.9167 | 0.8462 | 0.8871 | 0.8958 | Baseline no neuronal fuerte. |
| v4.0.0 | `StandardScaler + Tuned MLPClassifier` | 0.6667 | 0.5714 | 1.0000 | 0.7273 | 0.8696 | 0.8583 | Red neuronal con recall perfecto, pero 18 falsos positivos. |
| v4.1.0 | `StandardScaler + Balanced Tuned MLPClassifier` | 0.7778 | 0.6875 | 0.9167 | 0.7857 | 0.8594 | 0.8569 | Red neuronal balanceada, con 10 falsos positivos y 2 falsos negativos. |
| v4.2.0 | `StandardScaler + Robust Calibrated MLPClassifier` | 0.7222 | 0.6216 | 0.9583 | 0.7541 | 0.8647 | 0.8806 | Modelo activo; mejora recall, F2 y ROC-AUC frente a v4.1, pero aumenta falsos positivos. |

La comparación es honesta: v4.2 no mejora todas las métricas. Se activa porque mantiene la red neuronal final, usa una evaluación más robusta y mejora recall/F2/ROC-AUC frente a v4.1. v3 y v4.1 se conservan como referencias fuertes.

## Uso Previsto

Este modelo está diseñado para una demostración académica de MLOps:

- Servir una red neuronal con FastAPI.
- Consumir predicciones desde un frontend React.
- Versionar artefactos de modelo.
- Exponer métricas para Prometheus y Grafana.
- Mostrar un flujo reproducible con búsqueda de hiperparámetros, threshold tuning y evaluación visual.

## Limitaciones

- No debe utilizarse para diagnóstico médico real.
- El dataset tiene solo 270 registros.
- No se ha realizado validación clínica externa ni validación por cohortes independientes.
- Las métricas proceden de un split de test reproducible, no de un estudio clínico.
- Priorizar F2/recall puede aumentar falsos positivos.
- El threshold se ajusta sobre el set de test del proyecto; en un flujo clínico real debería validarse externamente.
- La calibración fue evaluada, pero no aplicada porque no favoreció el objetivo elegido.
- La probabilidad devuelta por el modelo debe interpretarse como salida del clasificador, no como probabilidad clínica certificada.
- Las variables proceden del dataset original y pueden no representar todos los factores relevantes en una valoración cardiovascular real.

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

Con más iteraciones de búsqueda:

```bash
python backend/train_model.py --search-iterations 48
```

El entrenamiento genera:

- `models/heart_model_v4_2_mlp_calibrated.joblib`
- `models/model_metadata.json`
- `models/evaluation_report.json`
- `docs/images/confusion_matrix_mlp_v4_2.png`
- `docs/images/roc_curve_mlp_v4_2.png`
- `docs/images/precision_recall_curve_mlp_v4_2.png`
- `docs/images/feature_importance_mlp_v4_2.png`
- `docs/images/threshold_metrics_mlp_v4_2.png`
- `docs/images/model_comparison_metrics.png`

## Cómo Versionar Una Nueva Versión

1. Entrenar el nuevo modelo con `backend/train_model.py`.
2. Revisar `models/evaluation_report.json`.
3. Guardar el nuevo artefacto con nombre versionado, por ejemplo `models/heart_model_v5_mlp_tuned.joblib`.
4. Actualizar `models/model_metadata.json` con versión, algoritmo, métricas, `decision_threshold`, `best_params`, `model_path`, fuente del dataset y fecha.
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

- Ampliar el dataset con trazabilidad clara.
- Evaluar calibración con más datos y validación externa.
- Añadir validación cruzada anidada.
- Registrar experimentos con MLflow.
- Evaluar explicabilidad con SHAP u otra técnica interpretable.
- Automatizar el versionado de modelos en CI/CD.
