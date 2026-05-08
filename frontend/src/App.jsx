import { useEffect, useMemo, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const GRAFANA_URL = import.meta.env.VITE_GRAFANA_URL || "http://localhost:3001";
const PROMETHEUS_URL = import.meta.env.VITE_PROMETHEUS_URL || "http://localhost:9090";

const initialPatient = {
  age: 63,
  sex: 1,
  chest: 3,
  resting_blood_pressure: 145,
  serum_cholestoral: 233,
  fasting_blood_sugar: 1,
  resting_electrocardiographic_results: 0,
  maximum_heart_rate_achieved: 150,
  exercise_induced_angina: 0,
  oldpeak: 2.3,
  slope: 1,
  number_of_major_vessels: 0,
  thal: 6
};

const formSections = [
  {
    title: "Datos del paciente",
    fields: [
      { name: "age", label: "Edad", type: "number", min: 1, max: 120, step: 1, unit: "años" },
      {
        name: "sex",
        label: "Sexo",
        type: "select",
        options: [
          { value: 0, label: "Mujer" },
          { value: 1, label: "Hombre" }
        ]
      }
    ]
  },
  {
    title: "Parámetros cardiovasculares",
    fields: [
      {
        name: "resting_blood_pressure",
        label: "Presión arterial en reposo",
        type: "number",
        min: 50,
        max: 250,
        step: 1,
        unit: "mmHg"
      },
      {
        name: "serum_cholestoral",
        label: "Colesterol sérico",
        type: "number",
        min: 100,
        max: 700,
        step: 1,
        unit: "mg/dl"
      },
      {
        name: "fasting_blood_sugar",
        label: "Glucosa en ayunas > 120 mg/dl",
        type: "select",
        options: [
          { value: 0, label: "No" },
          { value: 1, label: "Sí" }
        ]
      },
      {
        name: "resting_electrocardiographic_results",
        label: "ECG en reposo",
        type: "select",
        options: [
          { value: 0, label: "0: Normal" },
          { value: 1, label: "1: Anomalía ST-T" },
          { value: 2, label: "2: Hipertrofia ventricular probable" }
        ]
      },
      {
        name: "maximum_heart_rate_achieved",
        label: "Frecuencia cardíaca máxima",
        type: "number",
        min: 60,
        max: 230,
        step: 1,
        unit: "lpm"
      }
    ]
  },
  {
    title: "Síntomas y prueba de esfuerzo",
    fields: [
      {
        name: "chest",
        label: "Tipo de dolor torácico",
        type: "select",
        options: [
          { value: 1, label: "1: Angina típica" },
          { value: 2, label: "2: Angina atípica" },
          { value: 3, label: "3: Dolor no anginoso" },
          { value: 4, label: "4: Asintomático" }
        ]
      },
      {
        name: "exercise_induced_angina",
        label: "Angina inducida por ejercicio",
        type: "select",
        options: [
          { value: 0, label: "No" },
          { value: 1, label: "Sí" }
        ]
      },
      { name: "oldpeak", label: "Depresión ST", type: "number", min: 0, max: 10, step: 0.1 },
      {
        name: "slope",
        label: "Pendiente del segmento ST",
        type: "select",
        options: [
          { value: 1, label: "1: Ascendente" },
          { value: 2, label: "2: Plana" },
          { value: 3, label: "3: Descendente" }
        ]
      }
    ]
  },
  {
    title: "Variables clínicas adicionales",
    fields: [
      {
        name: "number_of_major_vessels",
        label: "Vasos principales",
        type: "select",
        options: [
          { value: 0, label: "0" },
          { value: 1, label: "1" },
          { value: 2, label: "2" },
          { value: 3, label: "3" }
        ]
      },
      {
        name: "thal",
        label: "Thal",
        type: "select",
        options: [
          { value: 3, label: "3: Normal" },
          { value: 6, label: "6: Defecto fijo" },
          { value: 7, label: "7: Defecto reversible" }
        ]
      }
    ]
  }
];

const fields = formSections.flatMap((section) => section.fields);

const inputSummary = [
  { name: "resting_blood_pressure", label: "Presión arterial", min: 50, max: 250, unit: "mmHg" },
  { name: "serum_cholestoral", label: "Colesterol", min: 100, max: 700, unit: "mg/dl" },
  { name: "maximum_heart_rate_achieved", label: "Frecuencia máxima", min: 60, max: 230, unit: "lpm" },
  { name: "oldpeak", label: "Depresión ST", min: 0, max: 10, unit: "" }
];

function getInitialTheme() {
  if (typeof window === "undefined") return "light";

  const storedTheme = window.localStorage.getItem("heart-ui-theme");
  if (storedTheme === "light" || storedTheme === "dark") return storedTheme;

  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function apiStatusLabel(status) {
  if (!status) return "Comprobando";
  if (status.status === "healthy") return "Operativa";
  if (status.status === "offline") return "Sin conexión";
  return "No disponible";
}

function riskLabel(level) {
  if (level === "High") return "Alto";
  if (level === "Medium") return "Medio";
  return "Bajo";
}

function predictionText(label) {
  return label === "Disease" ? "Posible enfermedad" : "Sin indicios de enfermedad";
}

function formatPercent(value) {
  if (value == null || Number.isNaN(Number(value))) return "No disponible";
  return `${Math.round(Number(value) * 100)}%`;
}

function percentValue(value) {
  if (value == null || Number.isNaN(Number(value))) return 0;
  return Math.round(Number(value) * 100);
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function rangePosition(value, min, max) {
  if (!Number.isFinite(Number(value)) || max <= min) return 0;
  return clamp(((Number(value) - min) / (max - min)) * 100, 0, 100);
}

function isValidFieldValue(field, value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) return false;
  if (field.options) return field.options.some((option) => option.value === numericValue);
  return numericValue >= field.min && numericValue <= field.max;
}

function FieldControl({ field, value, onChange }) {
  return (
    <label className="fieldControl">
      <span className="fieldLabel">{field.label}</span>
      <span className="fieldInputRow">
        {field.type === "select" ? (
          <select value={value} onChange={(event) => onChange(field.name, event.target.value)}>
            {field.options.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        ) : (
          <input
            type="number"
            min={field.min}
            max={field.max}
            step={field.step}
            value={value}
            onChange={(event) => onChange(field.name, event.target.value)}
          />
        )}
        {field.unit && <small>{field.unit}</small>}
      </span>
    </label>
  );
}

function SummaryCard({ label, value, detail, tone = "neutral" }) {
  return (
    <section className={`dashboardCard summaryCard tone-${tone}`}>
      <span className="cardEyebrow">{label}</span>
      <strong>{value}</strong>
      <p>{detail}</p>
    </section>
  );
}

function App() {
  const [theme, setTheme] = useState(getInitialTheme);
  const [patient, setPatient] = useState(initialPatient);
  const [result, setResult] = useState(null);
  const [lastPredictionAt, setLastPredictionAt] = useState(null);
  const [status, setStatus] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [versionInfo, setVersionInfo] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const requiredFieldsOk = useMemo(
    () => fields.every((field) => isValidFieldValue(field, patient[field.name])),
    [patient]
  );

  const probabilities = useMemo(() => {
    const disease = result?.probabilities?.disease ?? result?.probability ?? null;
    const noDisease = result?.probabilities?.no_disease ?? (disease != null ? 1 - disease : null);
    return { disease, noDisease };
  }, [result]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("heart-ui-theme", theme);
  }, [theme]);

  useEffect(() => {
    async function loadApiInfo() {
      try {
        const [healthResponse, infoResponse, versionResponse] = await Promise.all([
          fetch(`${API_URL}/health`),
          fetch(`${API_URL}/info`),
          fetch(`${API_URL}/version`)
        ]);
        if (healthResponse.ok) setStatus(await healthResponse.json());
        if (infoResponse.ok) setModelInfo(await infoResponse.json());
        if (versionResponse.ok) setVersionInfo(await versionResponse.json());
      } catch {
        setStatus({ status: "offline", model_loaded: false });
      }
    }

    loadApiInfo();
  }, []);

  function updateField(name, value) {
    setPatient((current) => ({
      ...current,
      [name]: value === "" ? "" : Number(value)
    }));
  }

  async function runPrediction() {
    setError("");

    if (!requiredFieldsOk) {
      setError("Revisa los valores del formulario antes de ejecutar la evaluación.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patient)
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "La API no pudo completar la evaluación.");
      }

      const payload = await response.json();
      setResult(payload);
      setLastPredictionAt(new Date());
    } catch (caughtError) {
      setError(caughtError.message);
    } finally {
      setLoading(false);
    }
  }

  async function submitPrediction(event) {
    event.preventDefault();
    await runPrediction();
  }

  const apiOnline = status?.status === "healthy";
  const activeModelVersion = versionInfo?.model_version || modelInfo?.version || status?.model_version || "No disponible";
  const activeAlgorithm = modelInfo?.algorithm || "StandardScaler + MLPClassifier";
  const diseasePercent = percentValue(probabilities.disease);
  const noDiseasePercent = percentValue(probabilities.noDisease);
  const nextTheme = theme === "dark" ? "light" : "dark";
  const lastPredictionText = lastPredictionAt
    ? lastPredictionAt.toLocaleString("es-ES", { dateStyle: "short", timeStyle: "medium" })
    : "Sin predicciones";
  const inferenceValue = result?.inference_time_ms != null ? `${result.inference_time_ms} ms` : "Pendiente";
  const resultLabel = result ? predictionText(result.label) : "Sin evaluación";
  const resultTone = result?.risk_level === "High" ? "danger" : result?.risk_level === "Medium" ? "warning" : "success";

  return (
    <main className="appShell">
      <header className="masthead">
        <div className="brandBlock">
          <div className="brandMark">
            <img src="/assets/logo-icon.png" alt="Heart Disease MLOps logo" />
          </div>
          <div>
            <p>Heart Disease MLOps</p>
            <h1>Evaluación de riesgo cardíaco</h1>
          </div>
        </div>

        <div className="mastheadActions" aria-label="Controles y estado del sistema">
          <span className={apiOnline ? "statusBadge statusOk" : "statusBadge statusWarn"}>
            API {apiStatusLabel(status)}
          </span>
          <span className="statusBadge">Modelo {activeModelVersion}</span>
          <nav className="technicalLinks" aria-label="Accesos técnicos">
            <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer">
              Swagger
            </a>
            <a href={`${API_URL}/metrics`} target="_blank" rel="noreferrer">
              Metrics
            </a>
            <a href={PROMETHEUS_URL} target="_blank" rel="noreferrer">
              Prometheus
            </a>
            <a href={GRAFANA_URL} target="_blank" rel="noreferrer">
              Grafana
            </a>
          </nav>
          <button
            className="themeToggle"
            type="button"
            aria-label={`Cambiar a modo ${nextTheme === "dark" ? "oscuro" : "claro"}`}
            aria-pressed={theme === "dark"}
            onClick={() => setTheme(nextTheme)}
          >
            <span aria-hidden="true" />
            {theme === "dark" ? "Modo oscuro" : "Modo claro"}
          </button>
        </div>
      </header>

      <div className="dashboardPage">
        <section className="summaryGrid" aria-label="Resumen del dashboard">
          <SummaryCard
            label="Estado del servicio"
            value={apiStatusLabel(status)}
            detail={`Modelo cargado: ${status?.model_loaded ? "sí" : "no"} · Entorno: ${versionInfo?.environment || "local"}`}
            tone={apiOnline ? "success" : "warning"}
          />
          <SummaryCard
            label="Modelo activo"
            value={activeModelVersion}
            detail={`${activeAlgorithm} · exactitud ${modelInfo?.accuracy != null ? formatPercent(modelInfo.accuracy) : "no disponible"}`}
          />
          <SummaryCard
            label="Última inferencia"
            value={inferenceValue}
            detail={lastPredictionText}
            tone={result ? resultTone : "neutral"}
          />
          <SummaryCard
            label="Resultado actual"
            value={resultLabel}
            detail={result ? `Riesgo ${riskLabel(result.risk_level)} · enfermedad ${formatPercent(probabilities.disease)}` : "Ejecuta una evaluación para ver el score"}
            tone={result ? resultTone : "neutral"}
          />
        </section>

        <section className="dashboardGrid">
          <section className="dashboardCard formCard" aria-labelledby="clinical-form-title">
            <div className="cardHeader">
              <div>
                <span className="cardEyebrow">Datos clínicos de entrada</span>
                <h2 id="clinical-form-title">Formulario de evaluación</h2>
              </div>
              <p>Los campos mantienen la codificación esperada por la API.</p>
            </div>

            <form className="clinicalForm" onSubmit={submitPrediction}>
              {formSections.map((section) => (
                <fieldset className="formSection" key={section.title}>
                  <legend>{section.title}</legend>
                  <div className="fieldGrid">
                    {section.fields.map((field) => (
                      <FieldControl
                        key={field.name}
                        field={field}
                        value={patient[field.name]}
                        onChange={updateField}
                      />
                    ))}
                  </div>
                </fieldset>
              ))}

              <p className="formNote">Las variables categóricas conservan la codificación del dataset original.</p>

              {error && (
                <div className="errorBanner" role="alert">
                  {error}
                </div>
              )}

              <div className="formActions">
                <button className="primaryButton" type="submit" disabled={loading || !requiredFieldsOk}>
                  {loading ? "Evaluando..." : "Evaluar riesgo"}
                </button>
                <span>{requiredFieldsOk ? "Formulario listo" : "Hay valores fuera de rango"}</span>
              </div>
            </form>
          </section>

          <aside className="resultColumn" aria-label="Resultado y visualizaciones">
            <section className="dashboardCard resultCard">
              <div className="cardHeader">
                <div>
                  <span className="cardEyebrow">Resultado del modelo</span>
                  <h2>Evaluación actual</h2>
                </div>
                <span className={`riskBadge risk-${result?.risk_level?.toLowerCase() || "pending"}`}>
                  {result ? `Riesgo ${riskLabel(result.risk_level)}` : "Pendiente"}
                </span>
              </div>

              {!result && !loading && (
                <div className="emptyResult" role="status">
                  <span className="emptyStateIcon" aria-hidden="true">
                    i
                  </span>
                  <strong>Sin evaluación todavía</strong>
                  <p>Completa el formulario y ejecuta una predicción para ver el resultado del modelo.</p>
                  <button className="secondaryButton" type="button" onClick={runPrediction} disabled={!requiredFieldsOk}>
                    Evaluar riesgo
                  </button>
                </div>
              )}

              {loading && (
                <div className="emptyResult" role="status">
                  <span className="emptyStateIcon loadingDot" aria-hidden="true">
                    ...
                  </span>
                  <strong>Evaluando caso</strong>
                  <p>La API está procesando la entrada y registrando métricas operativas.</p>
                </div>
              )}

              {result && (
                <div className="resultHero">
                  <span>Resultado principal</span>
                  <strong>{predictionText(result.label)}</strong>
                  <p>Probabilidad estimada de enfermedad: {formatPercent(probabilities.disease)}</p>
                  <dl className="resultMeta">
                    <div>
                      <dt>Inferencia</dt>
                      <dd>{result.inference_time_ms} ms</dd>
                    </div>
                    <div>
                      <dt>Modelo</dt>
                      <dd>{result.model_version}</dd>
                    </div>
                    <div>
                      <dt>Última predicción</dt>
                      <dd>{lastPredictionText}</dd>
                    </div>
                  </dl>
                </div>
              )}
            </section>

            <section className="dashboardCard probabilityCard">
              <div className="cardHeader compact">
                <div>
                  <span className="cardEyebrow">Probabilidades estimadas</span>
                  <h2>Distribución de salida</h2>
                </div>
              </div>
              {result ? (
                <>
                  <div className="probabilityBars">
                    <div>
                      <span>Enfermedad</span>
                      <div className="barTrack">
                        <i style={{ width: `${diseasePercent}%` }} />
                      </div>
                      <b>{diseasePercent}%</b>
                    </div>
                    <div>
                      <span>Sin enfermedad</span>
                      <div className="barTrack secondary">
                        <i style={{ width: `${noDiseasePercent}%` }} />
                      </div>
                      <b>{noDiseasePercent}%</b>
                    </div>
                  </div>
                  <div className="riskScale" aria-label="Nivel de riesgo">
                    {["Low", "Medium", "High"].map((level) => (
                      <span
                        key={level}
                        className={result?.risk_level === level ? `active segment-${level.toLowerCase()}` : ""}
                      >
                        {riskLabel(level)}
                      </span>
                    ))}
                  </div>
                </>
              ) : (
                <p className="mutedCardText">La distribución de salida aparecerá después de la primera evaluación.</p>
              )}
            </section>

            <section className="dashboardCard inputSnapshot">
              <div className="cardHeader compact">
                <div>
                  <span className="cardEyebrow">Variables clave</span>
                  <h2>Resumen de entrada</h2>
                </div>
              </div>
              <div className="snapshotRows">
                {inputSummary.map((item) => (
                  <div key={item.name}>
                    <span>{item.label}</span>
                    <div className="barTrack compactBar">
                      <i style={{ width: `${rangePosition(patient[item.name], item.min, item.max)}%` }} />
                    </div>
                    <b>
                      {patient[item.name]} {item.unit}
                    </b>
                  </div>
                ))}
              </div>
            </section>
          </aside>
        </section>
      </div>

      <footer className="appFooter">
        <p>Resultado orientativo para uso académico. No sustituye una valoración médica profesional.</p>
      </footer>
    </main>
  );
}

export default App;
