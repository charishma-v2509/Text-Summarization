const { spawn } = require("child_process");

const generateSummary = (req, res) => {
    const { text, length } = req.body;

    const pythonProcess = spawn("python", [
        "summarizer/extractive_inference.py"
    ]);

    let result = "";

    pythonProcess.stdin.write(
        JSON.stringify({ text, length })
    );
    pythonProcess.stdin.end();

    pythonProcess.stdout.on("data", (data) => {
        result += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
        console.error("Python error:", data.toString());
    });

    pythonProcess.on("close", () => {
        try {
            const parsed = JSON.parse(result);
            res.json(parsed);
        } catch (err) {
            res.status(500).json({ error: "Summarization failed" });
        }
    });
};

module.exports = { generateSummary };
