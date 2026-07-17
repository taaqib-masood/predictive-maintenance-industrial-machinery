# Dynamic Hybrid Predictive Maintenance System
### AI-Based Framework for Industrial Machinery Using GBM, LSTM, and CNN with Edge Deployment

## Overview

Unplanned equipment failures cause significant operational and financial losses in industrial environments.
This project proposes a **dynamic hybrid predictive maintenance framework** that integrates multiple
machine learning and deep learning models to accurately predict Remaining Useful Life (RUL) and detect
machine failures under varying operational conditions.

The system dynamically selects between Gradient Boosting Machines (GBM), Long Short-Term Memory (LSTM)
networks, and Convolutional Neural Networks (CNN) based on real-time confidence scoring and fault patterns.
The solution is further optimized for **edge deployment** using TensorFlow Lite, enabling low-latency
inference on industrial hardware.

## Research Paper

**Title:** PREDICTIVE MAINTENANCE OF INDUSTRIAL MACHINERY USING A DYNAMIC AI-BASED HYBRID MODEL

This repository contains the implementation, experiments, and deployment work
corresponding to the above research paper.



## Key Contributions

- Hybrid predictive maintenance architecture combining GBM, LSTM, and CNN models
- Dynamic meta-controller for adaptive model selection
- Time-domain and frequency-domain (FFT-based) feature analysis
- Remaining Useful Life (RUL) prediction under variable fault conditions
- Edge-optimized inference using TensorFlow Lite
- Designed for real-world industrial deployment


## Models Used

- **Gradient Boosting Machine (GBM)**  
  Models gradual and steady degradation trends using structured sensor data, providing
  stable early-stage predictions and feature importance insights.

- **Long Short-Term Memory (LSTM)**  
  Captures temporal dependencies in multivariate sensor time-series data and detects
  sudden nonlinear degradation patterns. Monte Carlo dropout is used for confidence estimation.

- **Convolutional Neural Network (CNN)**  
  Operates on FFT-transformed vibration and thermal signals to identify frequency-domain
  anomalies associated with mechanical faults such as cracks and misalignment.



## Meta-Controller Architecture

An adaptive meta-controller dynamically fuses predictions from the GBM, LSTM, and CNN models.
The controller assigns confidence-based weights to each model in real time, ensuring that the
most reliable model dominates under changing operational conditions.

Observed behavior includes:
- GBM dominance during early stable operation
- LSTM dominance during temporal degradation phases
- CNN dominance near failure when high-frequency anomalies emerge



## Technology Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy, Pandas
- TensorFlow Lite
- Matplotlib / Streamlit



## Datasets

The system was evaluated using benchmark and industrial datasets to ensure robustness
under realistic operating conditions:

- **CMAPSS Turbofan Engine Dataset**  
  Used to evaluate Remaining Useful Life (RUL) prediction performance and model
  generalization across different degradation trajectories.

- **Industrial Vibration and Multi-Sensor Datasets**  
  Includes vibration, temperature, and pressure signals used to simulate real-world
  machinery behavior and fault conditions.



## Project Structure

src/              Core model implementations and inference pipeline  
data/             Raw and processed datasets  
models/           Trained and optimized model files  
deployment/       Edge deployment and TensorFlow Lite optimization  
results/          Evaluation metrics and visualizations  
paper/            Research paper and documentation  


## Results

- Performance evaluation was conducted using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² metrics
- The proposed dynamic hybrid framework demonstrated improved robustness compared to individual models
- Time-domain (GBM, LSTM) and frequency-domain (CNN with FFT) models exhibited complementary strengths
- Strong correlation was observed between predicted and actual Remaining Useful Life (RUL)
- Residual analysis confirmed stable error distribution across varying operating conditions

- **Mean Absolute Error (MAE): 15.31**
- **Root Mean Squared Error (RMSE): 17.22**
- Strong correlation between predicted and actual RUL values
- Stable residual distribution across multiple fault scenarios


## Edge Deployment

To support real-time industrial applications, the trained models were quantized and converted
to TensorFlow Lite format. This enables low-latency inference on embedded edge devices such as
industrial controllers and single-board computers while maintaining predictive accuracy.


## Limitations

- Evaluation performed on benchmark and simulated industrial datasets
- Performance may vary under unseen real-world fault conditions
- Continuous retraining is required for long-term deployment stability


## Future Work

- Integration with real-time IIoT sensor streams
- Reinforcement learning for maintenance scheduling optimization
- Explainable AI dashboards for maintenance engineers
- Cloud–edge hybrid predictive maintenance architecture


## Author

**Taaqib Masood**  
B.Tech Computer Science and Engineering  
Specialization: Data Analytics  


Detailed quantitative results and model-wise performance comparisons are presented in the associated research paper.

<br><br>

---

## 👨‍💻 About the Author

<div align="center">

# <img src="https://readme-typing-svg.herokuapp.com?font=Inter&weight=600&size=30&pause=1000&color=10B981&center=true&vCenter=true&width=500&lines=Industrial+AI;Anomaly+Detection;Time-Series+Forecasting" alt="Typing SVG" />

<img src="https://capsule-render.vercel.app/api?type=waving&color=10B981&height=200&section=header&text=Taaqib%20Masood&fontSize=50&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Machine%20Learning%20Engineering&descAlignY=55&descAlign=50" />

<p align="center">
  <img src="https://img.shields.io/badge/Location-Global-10B981?style=for-the-badge&logo=google-maps&logoColor=white" />
  <img src="https://img.shields.io/badge/Education-Computer%20Science%20Engineer-000000?style=for-the-badge&logo=academia&logoColor=white" />
</p>

<p align="center">
  <a href="https://taaqib-portfolio.vercel.app/"><img src="https://img.shields.io/badge/Portfolio-10B981?style=for-the-badge&logo=vercel&logoColor=white" /></a>
  <a href="https://www.linkedin.com/in/taaqib-masood/"><img src="https://img.shields.io/badge/LinkedIn-000000?style=for-the-badge&logo=linkedin&logoColor=white" /></a>
  <a href="mailto:taaqibmasood@gmail.com"><img src="https://img.shields.io/badge/Email-10B981?style=for-the-badge&logo=gmail&logoColor=white" /></a>
  <a href="https://github.com/taaqib-masood"><img src="https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=white" /></a>
</p>

</div>

---

### 📖 About Me
I am a Software Engineer with a profound focus on AI/ML systems and Full Stack Development. I specialize in building enterprise-grade applications, robust distributed systems, and implementing scalable machine learning solutions in production environments. My engineering philosophy revolves around a strong product mindset, ensuring that the technology not only meets rigorous technical standards but also delivers exceptional user experiences. 

**Open To:** Senior Software Engineering roles, AI Engineer positions, and high-impact open-source contributions.

---

### ⚙️ Tech Stack

**Languages**
<p align="left">
  <a href="https://skillicons.dev"><img src="https://skillicons.dev/icons?i=py,ts,js,java,cpp,go,rust&theme=dark" /></a>
</p>

**Frontend & Backend**
<p align="left">
  <a href="https://skillicons.dev"><img src="https://skillicons.dev/icons?i=react,nextjs,tailwind,nodejs,postgres,mongodb,redis&theme=dark" /></a>
</p>

**Cloud & DevOps**
<p align="left">
  <a href="https://skillicons.dev"><img src="https://skillicons.dev/icons?i=aws,gcp,docker,kubernetes,githubactions&theme=dark" /></a>
</p>

---

### 🤖 AI / ML Expertise

| Domain | Proficiency | Details |
| :--- | :---: | :--- |
| **Large Language Models (LLMs)** | Advanced | Prompt Engineering, RAG Architectures, Agentic Systems, MCP |
| **Machine Learning** | Advanced | Predictive Modeling, Classification, Regression, Ensemble Methods |
| **Deep Learning** | Intermediate | Neural Networks, CNNs, NLP, PyTorch, TensorFlow |
| **MLOps** | Intermediate | Model Deployment, Monitoring, CI/CD for ML, Data Pipelines |

---

### 🏆 Featured Projects

- **[Taaqib Portfolio](https://github.com/taaqib-masood/Taaqib-Portfolio)**: Global Edge Delivery, 99+ Lighthouse Score.
- **[Predictive Maintenance](https://github.com/taaqib-masood/predictive-maintenance-industrial-machinery)**: High accuracy failure prediction on real-time sensor data.
- **[Salon Booking SaaS](https://github.com/taaqib-masood/salon-booking-saas)**: Multi-tenant architecture with secure Stripe payments.
- **[Majestic Constructions](https://github.com/taaqib-masood/majestic-constructions)**: High-traffic enterprise site with fast server-side rendering.

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=10B981&height=100&section=footer" />
</div>
