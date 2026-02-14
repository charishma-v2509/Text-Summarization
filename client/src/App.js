import React, { useState } from "react";
import SummaryForm from "./components/SummaryForm";
import "./App.css";

function App() {
  const [darkMode, setDarkMode] = useState(false);

  const toggleMode = () => {
    setDarkMode((prev) => !prev);
    document.body.classList.toggle("dark-mode", !darkMode);
  };

  return (
    <div className={`app ${darkMode ? "dark" : "light"}`}>
      <header className="header">
        <h1 className="title">
          <i className="icon fas fa-robot"></i> AI Text Summarizer
        </h1>
        <button onClick={toggleMode} className="toggle-btn">
          {darkMode ? (
            <>
              <i className="fas fa-sun"></i> Light Mode
            </>
          ) : (
            <>
              <i className="fas fa-moon"></i> Dark Mode
            </>
          )}
        </button>
      </header>
      <main className="main-content">
        <SummaryForm darkMode={darkMode} />
      </main>
      <footer className="footer">
        <p>Powered by T5 Transformer AI</p>
      </footer>
    </div>
  );
}

export default App;