# 🩺 MASH-Assist AI: Clinical Support Tool

**MASH-Assist AI** is a functional prototype developed as a portfolio project for the **MIT Hacking Medicine in São Paulo** hackathon. This tool is designed to "give a voice" to Metabolic Dysfunction-Associated Steatohepatitis (MASH), a silent but serious chronic disease, by tackling two of its biggest challenges: underdiagnosis and the lack of readily accessible clinical knowledge.

The project directly addresses **Track 1: MASH** and the **InterSystems GenAI Challenge**.

---

## ✨ Key Features

This project combines a classic machine learning model with a modern Retrieval-Augmented Generation (RAG) system to demonstrate a dual-function clinical support tool.

### 1. Risk Prediction Model
-   **Purpose:** To stratify a patient's risk of having MASH based on a range of common clinical and demographic data.
-   **Method:** An **XGBoost Classifier** model trained on the NHANES 2011-2018 dataset. The model's target variable is a proxy for MASH risk, where a Fatty Liver Index (FLI) score of >= 60 is classified as 'High Risk'.
-   **Input:** The model uses a core set of demographic, laboratory, examination, and questionnaire variables (e.g., age, gender, ethnicity, glucose, HbA1c, lipids, liver enzymes, blood pressure).
-   **Output:** A risk classification of **Low Risk** or **High Risk**.
-   **Implementation:** See `notebook_risk_prediction.ipynb` for the complete data processing, training, evaluation, and model interpretation using SHAP.

### 2. AI Knowledge Assistant
-   **Purpose:** To provide healthcare professionals with quick, accurate answers to questions about MASH diagnosis, management, and guidelines.
-   **Method:** A **Retrieval-Augmented Generation (RAG)** pipeline using Google's Gemini LLM.
-   **Knowledge Base:** The AI's knowledge is strictly limited to a curated set of PDF documents, ensuring answers are contextually relevant and accurate.
-   **Functionality:** Users can ask questions in natural language (e.g., "What are the key recommendations for the pharmacological treatment of MASH?") and receive a detailed answer synthesized from the source documents.
-   **Implementation:** See `notebook_ai_assistant_FAISS.ipynb` for the setup of the vector store and the question-answering chain.

---

## 🛠️ Technology Stack

-   **Backend & Modeling:** Python
-   **Machine Learning:** Scikit-learn, Pandas, NumPy, XGBoost, SHAP
-   **Generative AI:** LangChain, Google Gemini API (`gemini-1.5-flash`)
-   **Vector Store (Local):** FAISS
-   **Embeddings:** Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
-   **Development Environment:** Jupyter Notebook

---

## 🚀 How to Run the Project

Follow these steps to set up the project and run the notebooks locally.

### Prerequisites

-   Python 3.9+
-   A Google API Key for the Gemini model. You can get one from [Google AI Studio](https://aistudio.google.com/).

### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/MASH-Assist-AI.git](https://github.com/YOUR_USERNAME/MASH-Assist-AI.git)
cd MASH-Assist-AI
```

### 2. Set Up the Environment

Create and activate a virtual environment:

```bash
# Create the environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
# venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a file named `.env` in the root of the project directory and add your Google API key:

```
GOOGLE_API_KEY=YOUR_API_KEY_HERE
```

### 4. Run the Notebooks

Launch Jupyter Notebook or JupyterLab to explore the project:

```bash
# To start Jupyter Notebook
jupyter notebook
```

-   **To train the risk model:** Open and run the cells in `notebook_risk_prediction.ipynb`. This will process the raw data and save the trained model as `mash_risk_model.pkl`.
-   **To test the AI assistant:** Open and run the cells in `notebook_ai_assistant_FAISS.ipynb`. This will build the vector store (if it doesn't exist) and allow you to ask questions against the knowledge base.

---

## 📂 Project Structure

```
MASH-Assist-AI/
│
├── data/                     # Folder for raw NHANES data (.XPT files)
├── knowledge_base/           # Folder for PDF documents used by the RAG system
├── faiss_index/              # Saved FAISS vector store index
│
├── notebook_risk_prediction.ipynb  # Notebook for data processing and model training
├── notebook_ai_assistant_FAISS.ipynb # Notebook for the RAG AI Assistant
├── mash_risk_model.pkl       # Saved machine learning model
├── requirements.txt          # List of Python dependencies
├── .env                      # File for API keys (not committed to Git)
└── README.md                 # This file
```

---

## 🔮 Next Steps

-   **Develop a User Interface:** Build an interactive web application using **Streamlit** or Flask to host the risk calculator and AI assistant, making it accessible to end-users.
-   **Implement a Scalable Vector Database:** Replace the local FAISS index with a more robust and scalable vector database solution like **InterSystems IRIS**, Pinecone, or ChromaDB for production environments.
-   **Deploy the Application:** Package the models and application for deployment on a cloud service (e.g., AWS, Google Cloud, Heroku).
-   **Expand the Knowledge Base:** Incorporate a wider range of clinical guidelines, research papers, and medical literature to enhance the AI assistant's expertise.
-   **Refine the Prediction Model:** Experiment with different machine learning models or include more patient features to improve the accuracy and scope of the risk prediction.
