# 🛡️ BidSure-AI - AI Powered GeM Compliance & Collusion Detection Platform

> **Team CompilerCrew | Smart India Hackathon 2025**
> Government e-Marketplace (GeM) ke liye banaya gaya ek intelligent compliance engine.

### 🚀 Live Demo
**Deployed on Streamlit Cloud:** [Link will be added after deployment]

### 💡 Problem Statement
Manual verification of GeM tenders is highly time-consuming. Bidders often use the same PAN / Mobile to submit multiple bids, submit documents with missing signatures or stamps, and it becomes difficult for officers to find concrete proof. This increases the risk of cartel formation and procurement fraud.

### ✅ Our Solution - BidSure-AI
An end-to-end AI platform that automatically reads bidder documents to provide 10-point verification, intelligent risk scoring, interactive collusion graph, and bilingual voice summary - designed with an officer-first approach.

### ✨ Key Features
- **1. Multi-Format Ingestion:** Scanned, Stamped & Signed PDF support with OCR
- **2. 10-Doc AI Verification Engine:** PAN, GST, MSME, ITR, Experience, Turnover, Auth Letter, EMD, Make in India, Sign/Stamp
- **3. Compliance & Risk Engine:** Rule-based + ML Anomaly Detection
- **4. Trust Graph Visualization:** Interactive Network Graph for cartel proof
- **5. Explainable AI (SHAP):** Har risk score ke peeche ka reason evidence ke saath
- **6. Bilingual Voice Assistant:** English & Hindi Officer Summary

### 🛠️ Tech Stack
`Python | Streamlit | spaCy NLP | scikit-learn | PyMuPDF OCR | SHAP | NetworkX | PyVis | MySQL | React | gTTS`

### 📁 Project Structure
em-shield-proto/
├── app.py               # Main application logic
├── requirements.txt     # Dependencies
├── samples/             # Demo bidder documents
└── README.mdjavascript
### ⚙️ How to Run Locally
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

###🔒 Proprietary Note
Core detection algorithms, risk weighting logic and ML models are part of Team CompilerCrew's proprietary research. This repo contains deployment version for demonstration purpose.
Developed by Team CompilerCrew | Jalgaon, Maharashtra
CompilerCrew ![Our Team](./CompilerCrew/group_logo.png)
Bhavesh -> Team leader
Vidhi
Aishwarya
Pratik
Mohini
Shravani
