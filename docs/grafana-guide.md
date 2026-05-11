# Guía Rápida de Grafana

Esta guía explica cómo ver en Grafana las métricas del proyecto **Heart Disease MLOps**.

---

## 1. Abrir Grafana

Con los contenedores levantados, entra en:

```text
http://localhost:3001/login
```

Credenciales locales:

```text
usuario: admin
password: change-me-local-demo
```

Estas credenciales son solo para uso local.

---

## 2. Levantar el proyecto

Desde la carpeta principal del repositorio:

```bash
docker compose up --build
```

---

## 3. Generar datos para el dashboard

Grafana necesita que la API reciba predicciones para mostrar métricas.

Puedes generar datos con:

```bash
python scripts/demo_requests.py --requests 100 --sleep 0.01
```

También puedes generar datos usando el frontend en:

```text
http://localhost:3000
```

---

## 4. Abrir el dashboard

Dentro de Grafana, busca el dashboard:

```text
Heart Disease MLOps Observability
```

Ahí se pueden ver:

- estado de la API;
- número total de predicciones;
- predicciones por clase;
- niveles de riesgo;
- latencia;
- errores;
- información del modelo activo.

---

## 5. Si no aparecen datos

Comprueba lo siguiente:

1. Que la API esté funcionando:

```text
http://localhost:8000/health
```

2. Que Prometheus vea la API como `UP`:

```text
http://localhost:9090/targets
```

3. Que se hayan generado predicciones:

```bash
python scripts/demo_requests.py --requests 100 --sleep 0.01
```

4. Que el rango temporal de Grafana esté en `Last 1 hour`.
