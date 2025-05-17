
# 📊 Credit Card Application ML Analyzer

A user-friendly Streamlit app to analyze and predict various aspects of credit card applications using Machine Learning.

---

## 🔍 Features

- **Application Status Classification** – Predict whether an application is Approved, Rejected, or Pending.
- **Default Risk Prediction** – Predict if an applicant is likely to default using a classification model.
- **Processing Time Estimation** – Estimate the number of days required to process an application.
- **Delay Bucket Classification** – Classify applications into delay buckets (e.g., 0-30, 30-60 days) based on total delay.
- **Anomaly Detection** – Detect suspicious or unusual applications using Isolation Forest.
- **Data Visualization** – View distribution, correlation heatmaps, and user-selectable plots.

---

## 🧠 Machine Learning Models Used

- `RandomForestClassifier`
- `RandomForestRegressor`
- `IsolationForest`
- `LabelEncoder` for categorical encoding
- Trained using `scikit-learn`

---

## 📁 File Structure

```
credit-app-analyzer/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
└── README.md             # Project description (this file)
```

---

## 🚀 How to Run Locally

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/credit-app-analyzer.git
   cd credit-app-analyzer
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

---

## 🌐 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url)

---

## 📌 Sample Data Used

If no CSV is uploaded, the app uses built-in sample data with fields like:
- `age`, `salary`, `credit_score`
- `application_status`, `default_flag`
- `processing_days`, `document_submission_delay`
- `dpd_bucket` – classified using custom logic

---

## 📈 Visualizations Available

- Countplot for application status or delay buckets
- Correlation Heatmap
- Salary vs Credit Score scatter
- Salary distribution histogram
- Credit Utilization barplot

---

## 👨‍💻 Author

- **Your Name**
- [LinkedIn](https://www.linkedin.com/in/abhisekabhipsita/) • [GitHub](https://github.com/AbhisekAbhipsita)

---

