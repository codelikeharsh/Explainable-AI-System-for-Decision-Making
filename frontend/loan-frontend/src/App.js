import React, { useState } from "react";
import "./App.css";
import jsPDF from "jspdf";
const API_URL = process.env.REACT_APP_API_URL;


function App() {
  const [formData, setFormData] = useState({
    income_annum: "",
    loan_amount: "",
    loan_term: "",
    cibil_score: "",
    total_assets: ""
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [darkMode, setDarkMode] = useState(true);
  const [auditLog, setAuditLog] = useState([]);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: Number(e.target.value)
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {

        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
      });

      if (!response.ok) throw new Error();

      const data = await response.json();
      setResult(data);

      setAuditLog((prev) => [
        {
          timestamp: new Date().toLocaleString(),
          decision: data.decision,
          probability: data.approval_probability
        },
        ...prev
      ]);
    } catch {
      setError("Unable to evaluate application. Please check backend.");
    } finally {
      setLoading(false);
    }
  };

  const downloadPDF = () => {
    const doc = new jsPDF();

    doc.setFontSize(14);
    doc.text("Loan Decision Report", 20, 20);

    doc.setFontSize(11);
    doc.text(`Decision: ${result.decision}`, 20, 35);
    doc.text(
      `Approval Probability: ${Math.round(result.approval_probability * 100)}%`,
      20,
      45
    );

    doc.text("Explanation:", 20, 65);
    result.explanation.forEach((line, i) => {
      doc.text(`- ${line}`, 22, 75 + i * 8);
    });

    doc.text("Counterfactual Advice:", 20, 120);
    result.counterfactual_advice.forEach((line, i) => {
      doc.text(`- ${line}`, 22, 130 + i * 8);
    });

    doc.save("loan_decision_report.pdf");
  };

  return (
    <div className={darkMode ? "container dark" : "container"}>
      <div className="content">
        {/* Top bar */}
        <div className="top-bar">
          <div className="toggle" onClick={() => setDarkMode(!darkMode)}>
            {darkMode ? "Light Mode" : "Dark Mode"}
          </div>
        </div>

        <h2>Loan Decision Intelligence</h2>
        <p className="subtitle">
          Explainable AI system for transparent loan approvals
        </p>

        {/* Form */}
        <form onSubmit={handleSubmit}>
          <div className="form-grid">
            <input name="income_annum" placeholder="Annual Income" onChange={handleChange} required />
            <input name="loan_amount" placeholder="Loan Amount" onChange={handleChange} required />
            <input name="loan_term" placeholder="Loan Term" onChange={handleChange} required />
            <input name="cibil_score" placeholder="CIBIL Score" onChange={handleChange} required />
            <input name="total_assets" placeholder="Total Assets" onChange={handleChange} required />
          </div>

          <button type="submit">Evaluate Application</button>
        </form>

        {loading && <p className="loading">Evaluating application…</p>}
        {error && <p className="error">{error}</p>}

        {/* Result */}
        {result && (
          <div className="result">
            <div className="result-header">
              <div className="decision">Decision: {result.decision}</div>
              <span
                className={`badge ${
                  result.decision === "Approved" ? "approved" : "rejected"
                }`}
              >
                {result.decision}
              </span>
            </div>

            {/* Confidence bar */}
            <div className="confidence-container">
              <div className="confidence-label">
                Approval Confidence: {Math.round(result.approval_probability * 100)}%
              </div>
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{
                    width: `${result.approval_probability * 100}%`
                  }}
                />
              </div>
            </div>

            <div className="section-title">Explanation</div>
            <ul>
              {result.explanation.map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ul>

            <div className="section-title">Counterfactual Advice</div>
            <ul>
              {result.counterfactual_advice.map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ul>

            <button className="secondary-btn" onClick={downloadPDF}>
              Download Decision Report
            </button>
          </div>
        )}

        {/* Audit Log */}
        {auditLog.length > 0 && (
          <div className="result">
            <div className="section-title">Audit Log</div>
            <ul>
              {auditLog.map((log, i) => (
                <li key={i}>
                  [{log.timestamp}] — {log.decision} ({log.probability})
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
