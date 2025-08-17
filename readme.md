# 🩺 MASH-Assist AI: Clinical Support Tool

**MASH-Assist AI** is a functional prototype developed as a portfolio project for the **MIT Hacking Medicine in São Paulo** hackathon. This tool is designed to "give a voice" to Metabolic Dysfunction-Associated Steatohepatitis (MASH), a silent but serious chronic disease, by tackling two of its biggest challenges: underdiagnosis and the lack of readily accessible clinical knowledge.

The project directly addresses **Track 1: MASH** and the **InterSystems GenAI Challenge**.

---

## ✨ Key Features

This tool combines a classic machine learning model with a modern Retrieval-Augmented Generation (RAG) system to provide a dual-function clinical support dashboard.

### 1. Risk Prediction Calculator
-   **Purpose:** To quickly stratify a patient's risk of having advanced liver fibrosis based on simple, common lab results.
-   **Method:** Utilizes a pre-trained **Random Forest Classifier** model.
-   **Input:** Patient's Age, AST (U/L), ALT (U/L), and Platelet Count (x10^9/L).
-   **Output:** A calculated **FIB-4 Score**, a clear risk classification (**Low Risk** or **High Risk**), and a confidence score for the prediction.

### 2. AI Knowledge Assistant
-   **Purpose:** To provide healthcare professionals with quick, accurate answers to questions about MASH diagnosis, management, and guidelines.
-   **Method:** Implements a **Retrieval-Augmented Generation (RAG)** pipeline using Google's Gemini LLM.
-   **Knowledge Base:** The AI's knowledge is strictly limited to the official documents provided for the hackathon, ensuring answers are contextually relevant and accurate.
-   **Functionality:** Users can ask questions in natural language (e.g., "What is the prevalence of MASH in Latin America?") and receive a detailed answer synthesized from the source documents.

---

## 🛠️ Technology Stack

-   **Backend & Modeling:** Python
-   **Machine Learning:** Scikit-learn, Pandas, NumPy
-   **Generative AI:** LangChain, Google Gemini API (`gemini-1.5-flash`)
-   **Vector Store:** FAISS (for local development)
-   **Embeddings:** Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
-   **Web Framework:** Streamlit

---

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

-   Python 3.9+
-   A Google API Key for the Gemini model. You can get one from [Google AI Studio](https://aistudio.google.com/).

### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/MASH-Assist-AI.git](https://github.com/YOUR_USERNAME/MASH-Assist-AI.git)
cd MASH-Assist-AI
2. Set Up the EnvironmentCreate and activate a virtual environment:# Create the environment
python -m venv venv

# Activate it (on macOS/Linux)
source venv/bin/activate

# On Windows
# venv\Scripts\activate
Install the required dependencies:pip install -r requirements.txt
3. Configure API KeyCreate a file named .env in the root of the project directory and add your Google API key to it:GOOGLE_API_KEY=YOUR_API_KEY_HERE
4. Run the ApplicationExecute the following command in your terminal:streamlit run app.py
Your web browser will automatically open with the MASH-Assist AI application running.📂 Project StructureMASH-Assist-AI/
│
├── data/                     # Folder for raw NHANES data (.XPT files)
├── knowledge_base/           # Folder for PDF documents used by the RAG system
├── faiss_index/              # Saved FAISS vector store index
│
├── MASH-Assist-AI-Notebook.ipynb # Jupyter Notebook with data processing and model training
├── mash_risk_model.joblib    # Saved machine learning model
├── app.py                    # The main Streamlit application script
├── requirements.txt          # List of Python dependencies
├── .env                      # File for API keys (not committed to Git)
└── README.md                 # This file
