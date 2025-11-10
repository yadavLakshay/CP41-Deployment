# 🚀 CP41-Deployment — NeuroScan: Brain Tumor Detection API

This repository contains the **deployment-ready implementation** of the NeuroScan project from **SDS CP041 (Advanced Track)**.  
It serves a deep learning model via a FastAPI backend, containerized with Docker, and hosted on Render Cloud.

---

## 🌐 Live App
👉 **[Click here to try the app](https://neuroscan-api-u1kp.onrender.com)**  
Upload an MRI scan to detect brain tumor presence using the deployed EfficientNetB0 model.

---

## ⚙️ Tech Stack
- **FastAPI** for serving the REST API  
- **TensorFlow + Keras** for model inference  
- **Docker** for containerization  
- **Render Cloud** for hosting  
- **HTML + JS + CSS** frontend interface

---

## 🧩 Repository Structure
```
CP41-Deployment/
├── app/
│ ├── pycache/
│ ├── static/
│ │ ├── script.js
│ │ └── style.css
│ ├── templates/
│ │ └── index.html
│ ├── main.py
│ ├── preprocessing.py
│ ├── utils.py
│ └── validator.py
├── assets/
│ ├── best_efficientnetb0_fixed.keras
│ └── ood_stats.npz
├── scripts/
│ └── build_ood_stats.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## 🧠 Features
- 🎯 **Real-time tumor prediction** from MRI scans  
- 🔐 **OOD validation** for unseen or invalid inputs  
- 💡 **Intuitive drag-and-drop UI**  
- 🐳 **Containerized & portable**  
- ☁️ **Deployed on Render (Free Tier)**

---


## 🛠️ How to Run Locally
```bash
git clone https://github.com/yadavLakshay/CP41-Deployment.git
cd CP41-Deployment
pip install -r requirements.txt
docker build -t neuroscan-app .
docker run -p 8000:8000 neuroscan-app

Access the app locally at:  
👉 [http://localhost:8000](http://localhost:8000)

```
---

## 📦 Source Project
This deployment is based on my original submission to the **SDS CP041 – NeuroScan Advanced Track Challenge.**  
It includes the full workflow with **data preprocessing, model development, hyperparameter tuning, MLflow tracking, and API deployment.**

🔗 **Full Project Repository:**  
🔗 **[Main Repo](https://github.com/yadavLakshay/SDS-CP041-neuroscan/tree/main/advanced/submissions/team-members/lakshay-yadav)**


---

## 👨‍💻 Author
**Lakshay Yadav**
