<h1 align="center">AI Text Summarizer</h1>

<p align="center">
Full-Stack Machine Learning Web Application for Automatic Text Summarization
</p>

<hr/>

<h2>Project Overview</h2>

<p>
AI Text Summarizer is a full-stack web application that generates concise summaries from long textual content. 
The system implements a Transformer-based encoder architecture for extractive summarization and provides 
a production-ready deployment using React, Express.js, and PyTorch.
</p>

<p>
The application supports user-controlled summary length selection (short, medium, long) 
and is deployed with a scalable cloud architecture.
</p>

<hr/>

<h2>Live Deployment</h2>

<p><strong>Frontend (Vercel):</strong><br/>
https://text-summarizer-beryl.vercel.app/
</p>

<p><strong>Backend API (Render):</strong><br/>
https://text-summarizer-api-kydf.onrender.com/api/summarize
</p>

<hr/>

<h2>System Architecture</h2>

<ul>
  <li>Frontend: React.js</li>
  <li>Backend: Node.js + Express.js</li>
  <li>Machine Learning Model: PyTorch Transformer Encoder</li>
  <li>Tokenization: SentencePiece (Subword Tokenization)</li>
  <li>Deployment: Vercel (Frontend) + Render (Backend)</li>
</ul>

<hr/>

<h2>Key Features</h2>

<ul>
  <li>Transformer-based extractive summarization</li>
  <li>User-selectable summary length (short / medium / long)</li>
  <li>RESTful API integration</li>
  <li>Cloud deployment with scalable backend</li>
  <li>End-to-end ML pipeline (training → inference → production)</li>
</ul>

<hr/>

<h2>Model Details</h2>

<ul>
  <li>Architecture: Transformer Encoder</li>
  <li>Embedding Size: 512</li>
  <li>Multi-Head Attention: 8 heads</li>
  <li>Vocabulary Size: 16,000 (SentencePiece)</li>
  <li>Loss Function: Binary Cross-Entropy</li>
  <li>Optimizer: Adam</li>
  <li>Training Samples: 10,000 articles</li>
  <li>Evaluation Metrics:
    <ul>
      <li>Accuracy</li>
      <li>F1 Score</li>
    </ul>
  </li>
</ul>

<hr/>

<h2>Performance Metrics</h2>

<ul>
  <li>Training Accuracy: ~80%</li>
  <li>Training F1 Score: ~0.85</li>
  <li>Validation Accuracy: ~79%</li>
  <li>Validation F1 Score: ~0.84</li>
</ul>

<hr/>

<h2>Project Structure</h2>

<pre>
Text-Summarization/
│
├── backend/
│   ├── server.js
│   ├── app.js
│   ├── routes/
│   ├── controllers/
│   ├── summarizer/
│   ├── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│
├── model/
├── training/
├── tokenizer/
└── checkpoints/
</pre>

<hr/>

<h2>How It Works</h2>

<ol>
  <li>User submits text from React frontend.</li>
  <li>Express backend receives request.</li>
  <li>Node process calls Python inference script.</li>
  <li>Transformer model processes text.</li>
  <li>Top-ranked sentences are selected.</li>
  <li>Summary returned as JSON response.</li>
</ol>

<hr/>

<h2>Installation (Local Setup)</h2>

<h3>Backend</h3>

<pre>
cd backend
npm install
pip install -r requirements.txt
node server.js
</pre>

<h3>Frontend</h3>

<pre>
cd frontend
npm install
npm start
</pre>

<hr/>

<h2>Future Improvements</h2>

<ul>
  <li>Add ROUGE evaluation metrics</li>
  <li>Implement beam search for abstractive summarization</li>
  <li>Add bullet-point formatting option</li>
  <li>Improve sentence ranking mechanism</li>
</ul>

<hr/>

<h2>Author</h2>

<p>
Sai Charishma Varkuti<br/>
Computer Science Engineering (AI & ML)
</p>
