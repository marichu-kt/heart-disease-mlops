# Resumen del taller base

El ZIP `Taller_PipelineModeling.zip` se usó como punto de partida. La estructura encontrada fue:

```text
Taller/
├── api_heart.py
├── heart_model_wrapper.py
├── heart_disease_model.joblib
├── Ejercicio_Practico_Despliegue_ML.ipynb
├── requirements.txt
├── requirements-flexible.txt
├── Dockerfile
└── Dockerfile.test
```

Decisiones tomadas:

- `heart_disease_model.joblib` se conserva como `models/heart_model_v1_logistic.joblib`.
- La lógica de API, wrapper, predicción batch y métricas se reutilizó como referencia para el backend nuevo.
- El notebook indicaba que el dataset se descargaba desde OpenML (`heart-statlog`) y que, si fallaba, se creaba un dataset sintético.
- En este proyecto no se usa dataset sintético. Si no existe `data/heart.csv`, el entrenamiento intenta descargar `heart-statlog` desde OpenML.
- La estructura final se reorganizó en `backend/`, `frontend/`, `models/`, `monitoring/`, `tests/` y `docs/`.
