# Guía Rápida de Uso

Esta guía explica de forma sencilla cómo ejecutar y probar el proyecto **Heart Disease MLOps**.

---

## 1. Ejecutar el proyecto

Desde la carpeta principal del repositorio, ejecuta:

```bash
docker compose up --build
```

Cuando termine, estarán disponibles:

| Servicio | URL |
|---|---|
| Frontend | http://localhost:3000 |
| API / Swagger | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001/login |

---

## 2. Probar la aplicación

Abre el frontend:

```text
http://localhost:3000
```

Desde ahí se pueden introducir los datos de un paciente y pulsar **Evaluar riesgo** para obtener:

- predicción del modelo;
- nivel de riesgo;
- probabilidad estimada;
- tiempo de inferencia;
- versión del modelo usado.

---

## 3. Probar la API

La API se puede probar desde Swagger:

```text
http://localhost:8000/docs
```

Endpoints principales:

| Endpoint | Uso |
|---|---|
| `GET /health` | Comprueba si la API funciona. |
| `GET /info` | Muestra información del modelo. |
| `POST /predict` | Realiza una predicción. |
| `POST /predict/batch` | Realiza varias predicciones. |
| `GET /metrics` | Muestra métricas para Prometheus. |

---

## 4. Generar métricas

Para generar tráfico de prueba:

```bash
python scripts/demo_requests.py --requests 100 --sleep 0.01
```

Esto crea peticiones automáticas a la API para que Prometheus y Grafana tengan datos que mostrar.

---

## 5. Ver métricas en Prometheus

Abre:

```text
http://localhost:9090
```

Consulta útil:

```promql
heart_predictions_total
```

También puedes revisar si la API está activa en:

```text
http://localhost:9090/targets
```

El target `heart-api` debería aparecer como `UP`.

---

## 6. Ver dashboard en Grafana

Abre:

```text
http://localhost:3001/login
```

Credenciales locales:

```text
usuario: admin
password: change-me-local-demo
```

Dashboard:

```text
Heart Disease MLOps Observability
```

Ahí se pueden ver predicciones, errores, latencia y estado del modelo.
