# Imágenes Del Proyecto

Esta carpeta contiene la identidad visual, capturas reales y recursos de documentación del proyecto.

Identidad visual:

- `logo-icon.png`: icono del proyecto, usado también como favicon del frontend.
- `logo-full.png`: logo completo con nombre, usado en el README.

Capturas y recursos técnicos:

- `frontend-light.png`
- `frontend-dark.png`
- `swagger.png`
- `prometheus-targets.png`
- `prometheus-graph.png`
- `grafana-dashboard.png`
- `confusion_matrix.png`
- `confusion_matrix_mlp_v4.png`
- `confusion_matrix_mlp_v4_1.png`

No subas capturas falsas ni pantallas mockeadas. Si necesitas regenerarlas, levanta el proyecto con `docker compose up --build`, genera tráfico con `python scripts/demo_requests.py --requests 25 --sleep 0.05` y captura las URLs documentadas en `README.md`. La matriz de confusión activa de la MLP v4.1 se regenera con `python backend/train_model.py`.
