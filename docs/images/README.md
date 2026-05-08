# Capturas Del Proyecto

Esta carpeta contiene o espera capturas reales generadas desde la aplicación ejecutándose con Docker Compose.

Rutas esperadas:

- `frontend-light.png`
- `frontend-dark.png`
- `swagger.png`
- `prometheus-targets.png`
- `prometheus-graph.png`
- `grafana-dashboard.png`
- `confusion_matrix.png`

No subas capturas falsas ni pantallas mockeadas. Si necesitas regenerarlas, levanta el proyecto con `docker compose up --build`, genera tráfico con `python scripts/demo_requests.py --requests 25 --sleep 0.05` y captura las URLs documentadas en `README.md`. La matriz de confusión se regenera con `python backend/train_model.py`.
