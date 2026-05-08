# Dataset

El ZIP del taller no incluye un CSV con el dataset. El script `backend/train_model.py` queda preparado para:

1. Leer `data/heart.csv` si se añade posteriormente.
2. Usar `--data-path /ruta/dataset.csv` si se quiere indicar otra ruta.
3. Descargar `heart-statlog` desde OpenML, que es la fuente usada por el notebook base.

No se genera dataset sintético para evitar entrenar con datos inventados.
