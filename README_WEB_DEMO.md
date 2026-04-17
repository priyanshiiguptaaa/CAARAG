# 🌐 Web Interface for Confidence-Aware Adaptive RAG

This directory contains a **modern web-based demo** for the Confidence-Aware Adaptive RAG system, providing an interactive interface to visualize the adaptive retrieval process in real-time.

---

## 🎯 Features

- **Interactive Query Interface**: Submit questions and get answers with confidence scores
- **Real-Time Visualization**: Watch the adaptive retrieval loop in action
- **Signal Tracking**: Monitor Sr (Retrieval), Sl (LLM), and Sc (Consistency) signals across rounds
- **Adaptive Decision Insights**: See when and why the system expands k or reformulates queries
- **Beautiful UI**: Modern, responsive design with dark theme and smooth animations

---

## 📋 Prerequisites

Before running the web demo, ensure you have completed:

1. **Step 1**: Corpus preparation (`step1_corpus_preparation.py`)
2. **Step 2**: Index building (`step2_indexing.py`)

These steps generate the required data files:
- `data/corpus.json`
- `data/faiss_index.bin`
- `data/embeddings.npy`
- `data/id_map.pkl`
- `data/bm25.pkl`

---

## 🚀 Quick Start

### 1. Install Web Dependencies

```bash
pip install flask flask-cors
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### 2. Start the Web Server

```bash
python app.py
```

You should see:
```
✅ System initialized successfully!
Starting Flask web server on http://localhost:5000
```

### 3. Open in Browser

Navigate to: **http://localhost:5000**

---

## 🎨 Interface Overview

### Main Components

1. **Query Input Section**
   - Large text area for entering questions
   - Quick-access example queries
   - Submit button with loading states

2. **Final Answer Card**
   - Displays the final generated answer
   - Shows overall confidence score with color coding:
     - 🟢 Green (≥0.7): High confidence
     - 🟡 Yellow (0.4-0.7): Medium confidence
     - 🔴 Red (<0.4): Low confidence
   - Total rounds taken

3. **Adaptive Rounds Visualization**
   - Detailed breakdown of each retrieval round
   - Signal values (Sr, Sl, Sc) with progress bars
   - Current k value and query reformulations
   - Intermediate answers

4. **Confidence Signals Chart**
   - Bar chart showing signal evolution across rounds
   - Visual comparison of Sr, Sl, and Sc
   - Easy identification of bottlenecks

---

## 🔧 Configuration

The web interface uses the same `configs/config.yaml` file as the main system.

**Key Settings:**

```yaml
adaptive:
  max_rounds: 3              # Maximum retrieval rounds
  threshold_tau: 0.7         # Confidence threshold to stop
  delta_k: 5                 # k-expansion increment

generation:
  model: "mock"              # Use "mock" for fast demo
  n_samples: 3               # Multi-sampling for consistency

retrieval:
  top_k: 5                   # Initial retrieval count
  hybrid_method: "rrf"       # Reciprocal Rank Fusion
```

---

## 🧪 Testing the Demo

### Example Queries

Try these questions to see the adaptive loop in action:

**High Confidence (1 round):**
- "What year did World War II end?"
- "Who wrote the novel '1984'?"

**Medium Confidence (2-3 rounds):**
- "Who is the president of France?"
- "Where was Albert Einstein born?"

**Low Confidence (Multiple rounds with reformulation):**
- Complex multi-hop questions
- Questions requiring rare entities

---

## 📊 Understanding the Visualization

### Signal Interpretation

**Sr (Retrieval Confidence)**
- High: Retrieved documents are highly relevant
- Low: Poor document coverage → System expands k

**Sl (LLM Self-Confidence)**
- High: LLM is confident in its answer
- Low: Query-document misalignment → System reformulates query

**Sc (Consistency)**
- High: Multiple samples agree semantically
- Low: Answer ambiguity → System increases sampling

### Adaptive Actions

The system automatically:
1. **Expands k** when Sr < 0.5 (poor retrieval)
2. **Reformulates query** when Sl < 0.5 (LLM confusion)
3. **Increases sampling** when Sc < 0.5 (inconsistency)

---

## 🛠️ API Endpoints

The Flask backend exposes these REST APIs:

### `GET /api/status`
Check system readiness and configuration

**Response:**
```json
{
  "status": "ready",
  "config": {
    "max_rounds": 3,
    "threshold_tau": 0.7,
    "retrieval_model": "all-mpnet-base-v2",
    "llm_backend": "mock"
  }
}
```

### `POST /api/query`
Process a query through the adaptive RAG pipeline

**Request:**
```json
{
  "query": "Who is the president of France?"
}
```

**Response:**
```json
{
  "query": "Who is the president of France?",
  "final_answer": "Based on the documents...",
  "final_confidence": 0.85,
  "total_rounds": 2,
  "rounds": [
    {
      "round": 1,
      "k": 5,
      "signals": {"Sr": 0.65, "Sl": 0.70, "Sc": 0.80},
      "confidence": 0.72,
      "answer": "..."
    }
  ]
}
```

### `POST /api/retrieve`
Retrieve documents without generation (for testing)

**Request:**
```json
{
  "query": "Python programming",
  "top_k": 5
}
```

### `GET /api/examples`
Get example queries for quick testing

---

## 🐛 Troubleshooting

### "System not initialized" error

**Cause:** Missing data files from Step 1 or Step 2

**Solution:**
```bash
# Run corpus preparation
python step1_corpus_preparation.py

# Build indexes
python step2_indexing.py
```

### Port 5000 already in use

**Solution:** Change the port in `app.py`:
```python
app.run(host='0.0.0.0', port=8080, debug=False)
```

### Slow response times

**Causes:**
- Using real LLM (OpenAI/HuggingFace) instead of Mock
- Large corpus with CPU-only FAISS

**Solutions:**
- Use `model: "mock"` in config for instant responses
- Switch to `faiss-gpu` for faster retrieval
- Reduce `max_rounds` for quicker demos

---

## 🎓 For Presentations & Demos

### Best Practices

1. **Start with Mock LLM** for instant responses during live demos
2. **Use example queries** to showcase different confidence levels
3. **Highlight signal evolution** to explain the adaptive mechanism
4. **Show reformulation** to demonstrate diagnostic capabilities

### Demo Script

1. Open web interface
2. Submit a simple query → Show 1-round success
3. Submit a complex query → Show multi-round adaptation
4. Explain each signal (Sr, Sl, Sc) using the visualization
5. Point out when k-expansion or reformulation occurs

---

## 📁 File Structure

```
confidence_aware_rag/
├── app.py                    # Flask backend
├── templates/
│   └── index.html           # Main web interface
├── static/
│   ├── css/
│   │   └── style.css        # Styling
│   └── js/
│       └── app.js           # Frontend logic
└── README_WEB_DEMO.md       # This file
```

---

## 🔮 Future Enhancements

Potential additions for the web interface:

- [ ] Document viewer showing retrieved chunks
- [ ] Confidence calibration curve visualization
- [ ] Export results to JSON/PDF
- [ ] Query history and comparison
- [ ] Real-time streaming of adaptive rounds
- [ ] Integration with real LLMs (GPT-4, Claude)
- [ ] Batch query evaluation mode

---

## 📝 Citation

If you use this web interface in your research or presentations:

```bibtex
@article{yourname2026confidence,
  title={Confidence-Aware Adaptive Retrieval-Augmented Generation},
  author={Your Name},
  journal={Journal Name},
  year={2026}
}
```

---

## 📧 Support

For issues or questions about the web interface:
- Check the main README.md
- Review the troubleshooting section above
- Inspect browser console for JavaScript errors
- Check Flask logs for backend errors

---

**Enjoy exploring the Confidence-Aware Adaptive RAG system! 🚀**
