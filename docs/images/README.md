# Imágenes Del Proyecto

Esta carpeta contiene la identidad visual, capturas reales y recursos de documentación del proyecto.

Identidad visual:

- `logo-icon.png`: icono del proyecto, usado también como favicon del frontend.
- `logo-full.png`: logo completo con nombre, usado en el README.

Capturas y recursos técnicos:

- `frontend-light.png`
- `frontend-dark.png`
- `frontend-dashboard.png`
- `swagger.png`
- `prometheus-targets.png`
- `prometheus-graph.png`
- `grafana-dashboard.png`
- `confusion_matrix.png`
- `confusion_matrix_mlp_v4.png`
- `confusion_matrix_mlp_v4_1.png`
- `confusion_matrix_mlp_v4_2.png`
- `roc_curve_mlp_v4_2.png`
- `precision_recall_curve_mlp_v4_2.png`
- `feature_importance_mlp_v4_2.png`
- `threshold_metrics_mlp_v4_2.png`
- `model_comparison_metrics.png`

No subas capturas falsas ni pantallas mockeadas. Si necesitas regenerarlas, levanta el proyecto con `docker compose up --build`, genera tráfico con `python scripts/demo_requests.py --requests 500 --sleep 0.01` y captura las URLs documentadas en `README.md`. La matriz de confusión activa, curvas ROC/Precision-Recall, importancia por permutación, métricas por threshold y comparativa de modelos de la MLP v4.2 se regeneran con `python backend/train_model.py`.

Para capturas finales de observabilidad usa preferentemente:

```bash
python scripts/demo_requests.py --requests 500 --sleep 0.01
```

No se incluye `cv_metrics_boxplot.png` porque el reporte actual no almacena métricas completas por fold/repetición suficientes para construir un boxplot real sin inventar datos.
