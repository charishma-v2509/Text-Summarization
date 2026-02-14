import React, { useState } from "react";
import "./SummaryForm.css";

function SummaryForm({ darkMode }) {
  const [text, setText] = useState("");
  const [mode, setMode] = useState("medium");
  const [bulletPoints, setBulletPoints] = useState(false);
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
      const response = await fetch("http://localhost:5000/api/summarize", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({ text, mode, bullet_points: bulletPoints }),
      });

      if (!response.ok) {
        throw new Error("Failed to summarize text");
      }

      const data = await response.json();
      
      // Fix encoding issues
      let cleanedSummary = data.summary
        .replace(/â€TM/g, "'")  // Fix common encoding issues
        .replace(/â€œ/g, '"')   // Fix left double quote
        .replace(/â€/g, '"')    // Fix right double quote
        .replace(/â€™/g, "'");  // Fix apostrophe

      setSummary(cleanedSummary);
    } catch (err) {
      setError(err.message || "An error occurred while summarizing");
      setSummary("");
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(summary)
      .then(() => {
        alert("Summary copied to clipboard!");
      })
      .catch(() => {
        alert("Failed to copy summary");
      });
  };

  const formatBulletPoints = (text) => {
    return text.split('\n')
      .filter(line => line.trim() !== '')
      .map((line, index) => (
        <div key={index} className="bullet-point">
          • {line.trim()}
        </div>
      ));
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

        <div className="control-group">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={bulletPoints}
              onChange={(e) => setBulletPoints(e.target.checked)}
              className="checkbox-input"
            />
            <span>Bullet Points</span>
          </label>
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
          <div className="summary-output">
            {bulletPoints ? formatBulletPoints(summary) : summary}
          </div>
        </div>
      )}
    </div>
  );
}

export default SummaryForm;