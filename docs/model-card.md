# Model Card

## Modelo

Nombre del modelo:

```text
heart_disease_mlp_calibrated
```

Versión:

```text
v4.2.0
```

Algoritmo:

```text
StandardScaler + MLPClassifier
```

El modelo activo es una red neuronal multicapa (`MLPClassifier`) de scikit-learn. Se usa dentro de un `Pipeline` junto con `StandardScaler`.

---

## Objetivo

El modelo predice si un paciente tiene riesgo de enfermedad cardíaca según 13 variables clínicas.

La salida puede ser:

```text
0 = No Disease
1 = Disease
```

---

## Dataset

Se utilizó el dataset:

```text
OpenML heart-statlog
```

Este dataset tiene 270 registros. Es suficiente para una práctica académica, pero no para un sistema médico real.

---

## Variables de entrada

El modelo usa estas variables:

- edad;
- sexo;
- tipo de dolor torácico;
- presión arterial en reposo;
- colesterol;
- glucosa en ayunas;
- resultado del electrocardiograma;
- frecuencia cardíaca máxima;
- angina inducida por ejercicio;
- oldpeak;
- pendiente ST;
- número de vasos principales;
- thal.

---

## Métricas principales

| Métrica | Valor |
|---|---:|
| Accuracy | 0.7222 |
| Precision | 0.6216 |
| Recall | 0.9583 |
| F1-score | 0.7541 |
| F2-score | 0.8647 |
| ROC-AUC | 0.8806 |

La métrica más importante en este proyecto es el **F2-score**, porque da más peso al `recall`.

En este caso interesa detectar la mayoría de posibles casos `Disease`, aunque eso pueda generar más falsos positivos.

---

## Threshold de decisión

El threshold usado es:

```text
0.35
```

Regla usada por la API:

```text
probabilidad de Disease >= 0.35 -> Disease
probabilidad de Disease < 0.35  -> No Disease
```

---

## Matriz de confusión

```json
[
  [16, 14],
  [1, 23]
]
```

Lectura sencilla:

- 16 casos sanos fueron clasificados correctamente;
- 14 casos sanos fueron marcados como enfermedad;
- 1 caso con enfermedad no fue detectado;
- 23 casos con enfermedad fueron detectados correctamente.
