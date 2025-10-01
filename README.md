

RAG-Based Multi-Modal Generative AI Assistant

### 🌟 What It Is

Agent-Nesh is an **AI-powered personal assistant** that can understand and respond to **text, images, code, and voice**.
Unlike a simple chatbot, it uses **Retrieval-Augmented Generation (RAG)** and multiple specialized AI models to provide **context-aware, multimodal answers**.

Think of it like a **supercharged ChatGPT** that can also:
✅ Read and analyze images
✅ Help you debug/write code
✅ Understand spoken voice input
✅ Provide more factual and grounded responses using retrieval

---

# 🎯 Key Features

1. **Text Assistance** 📝

   * General Q&A, summaries, explanations
   * Works like ChatGPT but uses **RAG** to fetch external/contextual data

2. **Code Assistance** 💻

   * Debugging help
   * Code generation (Python, JS, etc.)
   * UML diagram → code conversion (via `uml_to_code.py` tool)

3. **Image Analysis** 🖼️

   * Upload an image → model analyzes and describes it
   * Can detect objects, scenes, and context

4. **Voice Recognition** 🎤

   * Converts spoken language to text using **OpenAI Whisper**
   * Lets you interact by speaking instead of typing

---

# 🧠 AI Models Used

* **Meta Llama 3** → General text generation & reasoning
* **Microsoft Phi 3 Vision** → Vision + text understanding (images & multimodal input)
* **IBM Granite** → Advanced NLP tasks and reasoning
* **OpenAI Whisper** → Speech-to-text for voice input

Each model is wrapped in Python scripts (`llama.py`, `phi_vision.py`, etc.) so the assistant can choose the right tool for the task.

---

# 🏛️ Project Structure (Simplified)

```
Generative agent/
├── models/                # Wrappers for AI models
├── chains/                # Task-specific chains (text, code, vision)
├── utils/                 # Utility functions (e.g., image processing)
├── agent/                 # Core agent logic
│   ├── tools/             # Tools like UML → code generator
│   └── llm_agent.py       # The main orchestrator
└── app.py                 # Streamlit app entry point
```

🔑 **Flow**:
User input → Chosen chain (language, code, vision) → Correct model → Response back via Streamlit UI.

---

# 🚀 How to Run It

1. **Clone repo**

   ```bash
   git clone https://github.com/ganeshnehru/RAG-Multi-Modal-Generative-AI-Agent.git
   cd RAG-Multi-Modal-Generative-AI-Agent
   ```

2. **Create virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate     # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** in `.env` file

   ```ini
   NVIDIA_API_KEY=your_nvidia_key
   OPENAI_API_KEY=your_openai_key
   ```

5. **Run app**

   ```bash
   streamlit run app.py
   ```

6. **Open browser** at `http://localhost:8501` → Interact with Agent-Nesh.

---

# 🛠️ Usage Examples

* **Ask a question** → *“Summarize this article”* → Llama 3 responds.
* **Coding help** → *“Write a Python function for binary search”* → Code assistant responds.
* **Upload an image** → *“What’s in this picture?”* → Phi Vision describes it.
* **Voice input** → Say *“What’s the weather in NYC today?”* → Whisper transcribes → Agent answers.

---

# 🔗 Tech Stack

* **Backend AI** → Llama 3, Phi Vision, Granite, Whisper
* **Framework** → Streamlit (for UI)
* **RAG** → Retrieval system for grounding answers
* **Tools** → Custom (UML → code, image processor)

---

# 🎯 Why It’s Useful

* Combines **multiple AI models** into one assistant
* Handles **multimodal input** (text, voice, image, code)
* Easy to run locally (just Python + Streamlit)
* Extensible → you can add more models/tools

---

👉 In short: **Agent-Nesh is like ChatGPT on steroids — a multi-modal, RAG-powered assistant that can chat, code, analyze images, and understand speech.**


