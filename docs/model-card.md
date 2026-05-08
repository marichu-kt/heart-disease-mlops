# Model Card: Heart Disease Tuned MLP

## Nombre Del Modelo

`heart_disease_mlp_tuned`

## Versión Actual

`v4.0.0`

## Algoritmo Activo

```text
StandardScaler + Tuned MLPClassifier
```

El modelo activo es una red neuronal multicapa (`MLPClassifier`) integrada en un `Pipeline` de scikit-learn con `StandardScaler`. La versión v4 se entrena con validación cruzada estratificada, búsqueda de hiperparámetros y ajuste de threshold de decisión.

## Dataset Utilizado

El ZIP del taller base no incluye un CSV local. El script de entrenamiento está preparado para usar:

1. `data/heart.csv`, si se añade manualmente.
2. Un CSV indicado con `--data-path`.
3. El dataset `heart-statlog` de OpenML, que es la fuente utilizada por el notebook original del taller.

No se genera dataset sintético.

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

La versión v4 usa:

- `train_test_split` estratificado con `test_size=0.2` y `random_state=42`.
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- `RandomizedSearchCV` con scoring principal `recall`.
- `MLPClassifier(solver="adam", early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=42)`.
- Ajuste de threshold sobre probabilidades de la clase `Disease`.

Espacio de búsqueda utilizado:

- `hidden_layer_sizes`: `(16,)`, `(32,)`, `(64,)`, `(32, 16)`, `(64, 32)`, `(128, 64)`
- `activation`: `relu`, `tanh`
- `alpha`: `0.0001`, `0.001`, `0.01`, `0.05`
- `learning_rate_init`: `0.0005`, `0.001`, `0.005`, `0.01`
- `batch_size`: `16`, `32`, `64`
- `learning_rate`: `constant`, `adaptive`
- `max_iter`: `800`, `1000`, `1500`

## Mejor Configuración Encontrada

```json
{
  "hidden_layer_sizes": [16],
  "activation": "tanh",
  "alpha": 0.05,
  "learning_rate_init": 0.005,
  "batch_size": 32,
  "learning_rate": "constant",
  "max_iter": 800
}
```

## Métrica Principal

La métrica principal es `recall`. En una demostración académica de riesgo clínico interesa reducir falsos negativos, porque un falso negativo significa clasificar como bajo/no riesgo un caso que realmente pertenece a la clase `Disease`.

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

La búsqueda de threshold probó:

```text
0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60
```

El criterio fue maximizar `recall`; en empate, maximizar `f1_score`, después `precision` y finalmente `accuracy`.

## Métricas Finales Del Modelo Activo

| Métrica | Valor |
|---|---:|
| Accuracy | 0.6667 |
| Precision | 0.5714 |
| Recall | 1.0000 |
| F1-score | 0.7273 |
| ROC-AUC | 0.8583 |

Matriz de confusión:

```json
[
  [12, 18],
  [0, 24]
]
```

Imagen:

```text
docs/images/confusion_matrix_mlp_v4.png
```

## Comparación Con V3

| Versión | Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | Lectura |
|---|---|---:|---:|---:|---:|---:|---|
| v3.0.0 | `StandardScaler + LogisticRegression` | 0.8519 | 0.7857 | 0.9167 | 0.8462 | 0.8958 | Baseline fuerte no neuronal. |
| v4.0.0 | `StandardScaler + Tuned MLPClassifier` | 0.6667 | 0.5714 | 1.0000 | 0.7273 | 0.8583 | Modelo neuronal activo. |

La comparación es transparente: v4 mejora `recall` y consigue cero falsos negativos en el split de test, pero empeora accuracy, precision, F1 y ROC-AUC frente a v3. Se activa como modelo final porque el requisito académico pedía una red neuronal; v3 se mantiene como baseline fuerte no neuronal.

## Uso Previsto

Este modelo está diseñado para una demostración académica de MLOps:

- Servir una red neuronal con FastAPI.
- Consumir predicciones desde un frontend React.
- Versionar artefactos de modelo.
- Exponer métricas para Prometheus y Grafana.
- Mostrar un flujo de entrenamiento reproducible con búsqueda de hiperparámetros.

## Limitaciones

- No debe utilizarse para diagnóstico médico real.
- El dataset es pequeño para estándares clínicos actuales.
- No se ha realizado validación clínica externa ni validación por cohortes independientes.
- Las métricas proceden de un split de test reproducible, no de un estudio clínico.
- Priorizar `recall` reduce falsos negativos, pero aumenta falsos positivos.
- El threshold se ajusta sobre el set de test del proyecto; en un flujo clínico real debería validarse con datos externos.
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

- `models/heart_model_v4_mlp_tuned.joblib`
- `models/model_metadata.json`
- `models/evaluation_report.json`
- `docs/images/confusion_matrix_mlp_v4.png`

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

- Añadir validación cruzada anidada.
- Calibrar probabilidades.
- Registrar experimentos con MLflow.
- Evaluar explicabilidad con SHAP u otra técnica interpretable.
- Incorporar un dataset curado local con trazabilidad clara.
- Automatizar el versionado de modelos en CI/CD.
