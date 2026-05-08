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
    description: "Variables demográficas básicas.",
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
    description: "Medidas clínicas registradas antes o durante la evaluación.",
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
    description: "Información asociada al dolor torácico y respuesta al ejercicio.",
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
    description: "Variables complementarias usadas por el dataset original.",
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
  const describedBy = `${field.name}-hint`;

  return (
    <label className="fieldControl">
      <span className="fieldLabel">{field.label}</span>
      <span className="fieldInputRow">
        {field.type === "select" ? (
          <select
            aria-describedby={describedBy}
            value={value}
            onChange={(event) => onChange(field.name, event.target.value)}
          >
            {field.options.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        ) : (
          <input
            aria-describedby={describedBy}
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
      <em id={describedBy}>
        {field.options ? "Valor codificado según el dataset" : `Rango admitido: ${field.min} - ${field.max}`}
      </em>
    </label>
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

  async function submitPrediction(event) {
    event.preventDefault();
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

  const apiOnline = status?.status === "healthy";
  const activeModelVersion = versionInfo?.model_version || modelInfo?.version || status?.model_version || "No disponible";
  const activeModelName = versionInfo?.model_name || modelInfo?.model_name || "heart_disease_mlp";
  const activeAlgorithm = modelInfo?.algorithm || "StandardScaler + MLPClassifier";
  const diseasePercent = percentValue(probabilities.disease);
  const noDiseasePercent = percentValue(probabilities.noDisease);
  const nextTheme = theme === "dark" ? "light" : "dark";
  const lastPredictionText = lastPredictionAt
    ? lastPredictionAt.toLocaleString("es-ES", { dateStyle: "short", timeStyle: "medium" })
    : "Sin predicciones";

  return (
    <main className="appShell">
      <header className="topBar">
        <div className="brandBlock">
          <div className="brandMark" aria-hidden="true">
            HD
          </div>
          <div>
            <p className="productLabel">Heart Disease MLOps</p>
            <h1>Evaluación de riesgo cardíaco</h1>
          </div>
        </div>
        <div className="headerControls" aria-label="Controles y estado del sistema">
          <div className="headerStatus" aria-label="Estado resumido del sistema">
            <span className={apiOnline ? "statusDot statusDotOk" : "statusDot statusDotWarn"} />
            <span>API {apiStatusLabel(status)}</span>
            <span>Modelo {activeModelVersion}</span>
            <span>{activeAlgorithm}</span>
          </div>
          <nav className="topLinks" aria-label="Accesos técnicos principales">
            <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer">
              Swagger
            </a>
            <a href={`${API_URL}/metrics`} target="_blank" rel="noreferrer">
              Metrics
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

      <section className="dashboardGrid">
        <section className="clinicalPanel" aria-labelledby="clinical-form-title">
          <div className="panelHeader">
            <div>
              <p className="sectionKicker">Datos clínicos de entrada</p>
              <h2 id="clinical-form-title">Formulario de evaluación</h2>
            </div>
            <p>Los campos mantienen la codificación esperada por la API para que la inferencia sea reproducible.</p>
          </div>

          <form className="clinicalForm" onSubmit={submitPrediction}>
            {formSections.map((section) => (
              <fieldset className="formSection" key={section.title}>
                <legend>
                  <span>{section.title}</span>
                  <small>{section.description}</small>
                </legend>
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

            {error && (
              <div className="errorBanner" role="alert">
                {error}
              </div>
            )}

            <div className="formActions">
              <button className="primaryButton" type="submit" disabled={loading || !requiredFieldsOk}>
                {loading ? "Evaluando..." : "Evaluar riesgo"}
              </button>
              <span>{requiredFieldsOk ? "Formulario listo para inferencia" : "Hay valores fuera de rango"}</span>
            </div>
          </form>
        </section>

        <aside className="insightsColumn" aria-label="Estado del sistema y resultado del modelo">
          <section className="systemPanel">
            <div className="panelHeader compact">
              <div>
                <p className="sectionKicker">Estado del servicio</p>
                <h2>Resumen operativo</h2>
              </div>
            </div>
            <dl className="definitionList">
              <div>
                <dt>API</dt>
                <dd>{apiStatusLabel(status)}</dd>
              </div>
              <div>
                <dt>Modelo cargado</dt>
                <dd>{status?.model_loaded ? "Sí" : "No"}</dd>
              </div>
              <div>
                <dt>Entorno</dt>
                <dd>{versionInfo?.environment || "local"}</dd>
              </div>
            </dl>
          </section>

          <section className="systemPanel">
            <h2>Modelo activo</h2>
            <dl className="definitionList">
              <div>
                <dt>Nombre</dt>
                <dd>{activeModelName}</dd>
              </div>
              <div>
                <dt>Versión</dt>
                <dd>{activeModelVersion}</dd>
              </div>
              <div>
                <dt>Exactitud</dt>
                <dd>{modelInfo?.accuracy != null ? formatPercent(modelInfo.accuracy) : "No disponible"}</dd>
              </div>
            </dl>
          </section>

          <section className="systemPanel">
            <h2>Accesos técnicos</h2>
            <nav className="linkList" aria-label="Accesos técnicos">
              <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer">
                Swagger
              </a>
              <a href={`${API_URL}/metrics`} target="_blank" rel="noreferrer">
                Métricas API
              </a>
              <a href={PROMETHEUS_URL} target="_blank" rel="noreferrer">
                Prometheus
              </a>
              <a href={GRAFANA_URL} target="_blank" rel="noreferrer">
                Grafana
              </a>
            </nav>
          </section>

          <section className="systemPanel">
            <h2>Última inferencia</h2>
            <p className="mutedText">{lastPredictionText}</p>
          </section>

          <section className="resultPanel" aria-label="Resultado del modelo">
            <div className="panelHeader compact">
              <div>
                <p className="sectionKicker">Resultado del modelo</p>
                <h2>Evaluación de riesgo</h2>
              </div>
              <span className="modelBadge">{activeModelVersion}</span>
            </div>

            {!result && !loading && (
              <div className="emptyResult">
                <strong>Sin evaluación todavía</strong>
                <p>Completa el formulario y ejecuta una predicción para ver el resultado, las probabilidades y el tiempo de inferencia.</p>
              </div>
            )}

            {loading && (
              <div className="emptyResult">
                <strong>Evaluando caso</strong>
                <p>La API está procesando la entrada y registrando métricas operativas.</p>
              </div>
            )}

            {result && (
              <div className="resultStack">
                <section className={`outcomeBox risk-${result.risk_level?.toLowerCase()}`}>
                  <span>Resultado final</span>
                  <strong>{predictionText(result.label)}</strong>
                  <p>Etiqueta del modelo: {result.label}</p>
                </section>

                <section className="riskScale" aria-label="Nivel de riesgo">
                  <div className="riskScaleHeader">
                    <span>Nivel de riesgo</span>
                    <strong>{riskLabel(result.risk_level)}</strong>
                  </div>
                  <div className="riskSegments">
                    {["Low", "Medium", "High"].map((level) => (
                      <span
                        key={level}
                        className={result.risk_level === level ? `active segment-${level.toLowerCase()}` : ""}
                      >
                        {riskLabel(level)}
                      </span>
                    ))}
                  </div>
                </section>

                <section className="probabilityPanel">
                  <div className="metricHeader">
                    <span>Probabilidad estimada</span>
                    <strong>{formatPercent(probabilities.disease)}</strong>
                  </div>
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
                </section>

                <section className="inputSnapshot">
                  <div className="metricHeader">
                    <span>Resumen visual de entrada</span>
                    <strong>Rangos del formulario</strong>
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

                <dl className="resultDetails">
                  <div>
                    <dt>Tiempo de inferencia</dt>
                    <dd>{result.inference_time_ms} ms</dd>
                  </div>
                  <div>
                    <dt>Versión del modelo</dt>
                    <dd>{result.model_version}</dd>
                  </div>
                  <div>
                    <dt>Última predicción</dt>
                    <dd>{lastPredictionText}</dd>
                  </div>
                </dl>
              </div>
            )}

            <p className="academicNotice">
              Resultado orientativo para uso académico. No sustituye una valoración médica profesional.
            </p>
          </section>
        </aside>
      </section>
    </main>
  );
}

export default App;
