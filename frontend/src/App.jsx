import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Gauge,
  HeartPulse,
  Loader2,
  Server,
  ShieldCheck,
  Sparkles,
  Stethoscope
} from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

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

const fields = [
  { name: "age", label: "Edad", min: 1, max: 120, step: 1, unit: "años" },
  { name: "sex", label: "Sexo", min: 0, max: 1, step: 1, hint: "0 mujer, 1 hombre" },
  { name: "chest", label: "Dolor torácico", min: 1, max: 4, step: 1 },
  { name: "resting_blood_pressure", label: "Presión arterial", min: 50, max: 250, step: 1, unit: "mmHg" },
  { name: "serum_cholestoral", label: "Colesterol", min: 100, max: 700, step: 1, unit: "mg/dl" },
  { name: "fasting_blood_sugar", label: "Glucosa en ayunas", min: 0, max: 1, step: 1, hint: "0 no, 1 sí" },
  { name: "resting_electrocardiographic_results", label: "ECG en reposo", min: 0, max: 2, step: 1 },
  { name: "maximum_heart_rate_achieved", label: "Frecuencia máxima", min: 60, max: 230, step: 1, unit: "lpm" },
  { name: "exercise_induced_angina", label: "Angina por ejercicio", min: 0, max: 1, step: 1, hint: "0 no, 1 sí" },
  { name: "oldpeak", label: "Oldpeak", min: 0, max: 10, step: 0.1 },
  { name: "slope", label: "Pendiente ST", min: 1, max: 3, step: 1 },
  { name: "number_of_major_vessels", label: "Vasos principales", min: 0, max: 3, step: 1 },
  { name: "thal", label: "Thal", min: 3, max: 7, step: 1 }
];

function riskClass(level) {
  if (level === "High") return "riskHigh";
  if (level === "Medium") return "riskMedium";
  return "riskLow";
}

function riskLabel(level) {
  if (level === "High") return "Riesgo alto";
  if (level === "Medium") return "Riesgo medio";
  return "Riesgo bajo";
}

function predictionLabel(label) {
  return label === "Disease" ? "Posible enfermedad" : "Sin enfermedad detectada";
}

function apiStatusLabel(status) {
  if (!status) return "Comprobando";
  if (status.status === "healthy") return "API operativa";
  if (status.status === "offline") return "API sin conexión";
  return "API no disponible";
}

function App() {
  const [patient, setPatient] = useState(initialPatient);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const requiredFieldsOk = useMemo(
    () =>
      fields.every((field) => {
        const value = Number(patient[field.name]);
        return Number.isFinite(value) && value >= field.min && value <= field.max;
      }),
    [patient]
  );

  useEffect(() => {
    async function loadApiInfo() {
      try {
        const [healthResponse, infoResponse] = await Promise.all([
          fetch(`${API_URL}/health`),
          fetch(`${API_URL}/info`)
        ]);
        if (healthResponse.ok) setStatus(await healthResponse.json());
        if (infoResponse.ok) setModelInfo(await infoResponse.json());
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
    setResult(null);

    if (!requiredFieldsOk) {
      setError("Revisa los valores del formulario antes de lanzar la predicción.");
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
        throw new Error(payload.detail || "La API no pudo generar la predicción.");
      }

      setResult(await response.json());
    } catch (caughtError) {
      setError(caughtError.message);
    } finally {
      setLoading(false);
    }
  }

  const diseaseProbability = result?.probability != null ? Math.round(result.probability * 100) : null;
  const apiOnline = status?.status === "healthy";

  return (
    <main className="appShell">
      <section className="heroBand">
        <div className="heroContent">
          <div className="brandRow">
            <div className="brandMark">
              <HeartPulse size={30} />
            </div>
            <span>Heart Disease MLOps</span>
          </div>
          <div className="heroGrid">
            <div>
              <p className="eyebrow">FastAPI + React + Prometheus + Grafana</p>
              <h1>Predicción clínica con modelo MLP versionado</h1>
              <p className="heroCopy">
                Dashboard médico para consultar el riesgo estimado de enfermedad cardíaca,
                consumir una API real y mostrar la versión del modelo usado en inferencia.
              </p>
              <div className="heroActions">
                <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer" className="ghostButton">
                  <Server size={18} />
                  Swagger
                </a>
                <a href={`${API_URL}/metrics`} target="_blank" rel="noreferrer" className="ghostButton">
                  <Gauge size={18} />
                  Métricas
                </a>
              </div>
            </div>
            <div className="statusPanel" aria-label="Estado del sistema">
              <div className={apiOnline ? "statusOk" : "statusWarn"}>
                <span className="panelLabel">API</span>
                <strong>{apiStatusLabel(status)}</strong>
                <small>Endpoint /health</small>
              </div>
              <div className={status?.model_loaded ? "statusOk" : "statusWarn"}>
                <span className="panelLabel">Modelo</span>
                <strong>{modelInfo?.version || status?.model_version || "..."}</strong>
                <small>{status?.model_loaded ? "Modelo cargado" : "Pendiente de carga"}</small>
              </div>
              <div className="statusNeutral">
                <span className="panelLabel">Algoritmo</span>
                <strong>{modelInfo?.algorithm || "MLPClassifier"}</strong>
                <small>Pipeline versionado</small>
              </div>
              <div className="statusNeutral">
                <span className="panelLabel">Exactitud</span>
                <strong>{modelInfo?.accuracy ? `${Math.round(modelInfo.accuracy * 100)}%` : "pendiente"}</strong>
                <small>Según metadatos</small>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="workspace">
        <form className="formPanel" onSubmit={submitPrediction}>
          <div className="sectionHeader">
            <div>
              <p className="eyebrow">Entrada del paciente</p>
              <h2>Variables clínicas</h2>
            </div>
            <Stethoscope size={28} />
          </div>

          <div className="fieldGrid">
            {fields.map((field) => (
              <label key={field.name} className="inputGroup">
                <span>{field.label}</span>
                <div className="inputShell">
                  <input
                    type="number"
                    min={field.min}
                    max={field.max}
                    step={field.step}
                    value={patient[field.name]}
                    onChange={(event) => updateField(field.name, event.target.value)}
                  />
                  {field.unit && <small>{field.unit}</small>}
                </div>
                <em>{field.hint || `${field.min} - ${field.max}`}</em>
              </label>
            ))}
          </div>

          {error && (
            <div className="errorBanner" role="alert">
              <AlertTriangle size={18} />
              {error}
            </div>
          )}

          <button className="primaryButton" type="submit" disabled={loading}>
            {loading ? <Loader2 className="spin" size={20} /> : <Sparkles size={20} />}
            {loading ? "Analizando..." : "Lanzar predicción"}
          </button>
        </form>

        <aside className="resultPanel">
          <div className="sectionHeader">
            <div>
              <p className="eyebrow">Resultado</p>
              <h2>Riesgo estimado</h2>
            </div>
            <Activity size={28} />
          </div>

          {!result && !loading && (
            <div className="emptyState">
              <ShieldCheck size={42} />
              <p>Completa el formulario y lanza una predicción para ver el resultado del modelo.</p>
            </div>
          )}

          {loading && (
            <div className="emptyState">
              <Loader2 className="spin" size={42} />
              <p>Consultando la API FastAPI y registrando métricas de inferencia.</p>
            </div>
          )}

          {result && (
            <div className={`resultCard ${riskClass(result.risk_level)}`}>
              <div className="resultTopline">
                <span>{predictionLabel(result.label)}</span>
                <CheckCircle2 size={22} />
              </div>
              <strong>{riskLabel(result.risk_level)}</strong>
              <p>Nivel de riesgo devuelto por el modelo versionado.</p>

              <div className="probabilityWrap">
                <span className="probabilityLabel">Probabilidad estimada de enfermedad</span>
                <div className="probabilityTrack">
                  <span style={{ width: `${diseaseProbability ?? 0}%` }} />
                </div>
                <b>{diseaseProbability != null ? `${diseaseProbability}%` : "Sin probabilidad"}</b>
              </div>

              <div className="resultStats">
                <div>
                  <span>Predicción</span>
                  <strong>{result.prediction}</strong>
                </div>
                <div>
                  <span>Modelo</span>
                  <strong>{result.model_version}</strong>
                </div>
                <div>
                  <span>Inferencia</span>
                  <strong>{result.inference_time_ms} ms</strong>
                </div>
              </div>
            </div>
          )}

          <div className="apiHint">
            <Server size={18} />
            <span>{API_URL}</span>
          </div>
        </aside>
      </section>
    </main>
  );
}

export default App;
