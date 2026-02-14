import React, { useState } from "react";
import "./SummaryForm.css";

function SummaryForm({ darkMode }) {
  const [text, setText] = useState("");
  const [mode, setMode] = useState("medium");
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSummarize = async () => {
    if (!text.trim()) {
      setError("Please enter some text to summarize");
      return;
    }
    if (text.trim().length < 100) {
      setError("Please enter at least 100 characters for better results");
      return;
    }

    setError("");
    setLoading(true);

    try {
      const response = await fetch(
        "https://text-summarizer-api-kydf.onrender.com/api/summarize",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify({ text, mode }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to summarize text");
      }

      const data = await response.json();

      let cleanedSummary = data.summary
        .replace(/â€TM/g, "'")
        .replace(/â€œ/g, '"')
        .replace(/â€/g, '"')
        .replace(/â€™/g, "'");

      setSummary(cleanedSummary);
    } catch (err) {
      setError(err.message || "An error occurred while summarizing");
      setSummary("");
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    navigator.clipboard
      .writeText(summary)
      .then(() => {
        alert("Summary copied to clipboard!");
      })
      .catch(() => {
        alert("Failed to copy summary");
      });
  };

  return (
    <div className={`form-container ${darkMode ? "dark" : ""}`}>
      <div className="input-section">
        <textarea
          rows={10}
          placeholder="Paste your text here (minimum 100 characters)..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="text-input"
        />
        <div className="char-count">{text.length} characters</div>
      </div>

      <div className="controls">
        <div className="control-group">
          <label>Summary Length:</label>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value)}
            className="select-input"
          >
            <option value="short">Short</option>
            <option value="medium">Medium</option>
            <option value="long">Long</option>
          </select>
        </div>

        <button
          onClick={handleSummarize}
          disabled={loading || text.length < 100}
          className="submit-btn"
        >
          {loading ? "Summarizing..." : "Generate Summary"}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {summary && (
        <div className="output-section">
          <div className="output-header">
            <h3>Summary Result</h3>
            <button onClick={handleCopy} className="copy-btn">
              Copy
            </button>
          </div>
          <div className="summary-output">{summary}</div>
        </div>
      )}
    </div>
  );
}

export default SummaryForm;
