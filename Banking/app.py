# import streamlit as st
# import pandas as pd
# import numpy as np
# from io import StringIO
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
# from sklearn.metrics import classification_report, mean_absolute_error, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# st.set_page_config(page_title="Credit App Analyzer", layout="wide")
# st.title("📊 Credit Card Application ML Analyzer")

# # -------------------------------
# # Sample fallback CSV with delay_bucket
# sample_csv = """
# customer_id,age,salary,credit_score,employment_type,application_source,document_completeness,past_payment_delays,credit_utilization,document_submission_delay,processing_days,application_status,default_flag,delay_bucket
# C001,35,60000,700,Salaried,Web,0.9,2,0.4,5,10,Approved,0,0-30
# C002,42,80000,720,Self-Employed,Agent,0.8,0,0.3,2,5,Approved,0,0-30
# C003,28,40000,650,Salaried,Web,0.7,5,0.5,10,15,Rejected,1,0-30
# C004,50,95000,800,Salaried,Branch,1.0,1,0.2,1,3,Approved,0,0-30
# C005,38,50000,620,Self-Employed,Agent,0.6,3,0.6,7,12,Pending,1,30-60
# """


# # ---------- CSV Selection Option ----------
# st.header("Step 1: Choose Your Data")
# use_sample = st.radio("Choose a data source:", ("📂 Upload your CSV", "📄 Use built-in sample CSV"))

# if use_sample == "📂 Upload your CSV":
#     uploaded_file = st.file_uploader("Upload your credit_card_application_data.csv", type=["csv"])
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file)
#         st.success("✅ Uploaded CSV loaded successfully.")
#     else:
#         st.warning("⚠️ Please upload a CSV file to proceed.")
#         st.stop()
# else:
#     df = pd.read_csv(StringIO(sample_csv))
#     st.success("✅ Using built-in sample CSV data.")

# # ---------- Data Preprocessing ----------
# def assign_bucket(days):
#     if days <= 30: return '0-30'
#     elif days <= 60: return '30-60'
#     elif days <= 90: return '60-90'
#     elif days <= 120: return '90-120'
#     elif days <= 150: return '120-150'
#     elif days <= 180: return '150-180'
#     else: return '180+'

# df["total_delay"] = df["processing_days"] + df["document_submission_delay"]
# df["dpd_bucket"] = df["total_delay"].apply(assign_bucket)

# label_cols = ["employment_type", "application_source", "application_status", "dpd_bucket"]
# label_encoders = {}
# for col in label_cols:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     label_encoders[col] = le

# if st.checkbox("Show DataFrame"):
#     st.dataframe(df)

# # ---------- Task Selection ----------
# st.header("Step 2: Select Task to Perform")
# task = st.selectbox("Choose a functionality", [
#     "Select",
#     "📌 Application Status Classification",
#     "⚠️ Default Flag Prediction",
#     "⏳ Processing Days Prediction (Regression)",
#     "📦 Delay Bucket Classification",
#     "🚨 Anomaly Detection"
# ])

# # ---------- Task Execution ----------
# if task == "📌 Application Status Classification":
#     st.subheader("Application Status Classification")
#     X = df.drop(columns=["customer_id", "application_status", "default_flag", "dpd_bucket", "total_delay"])
#     y = df["application_status"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     st.text("Classification Report:")
#     st.text(classification_report(y_test, y_pred))

#     cm = confusion_matrix(y_test, y_pred)
#     fig, ax = plt.subplots()
#     sns.heatmap(cm, annot=True, fmt="d", ax=ax,
#                 xticklabels=label_encoders["application_status"].classes_,
#                 yticklabels=label_encoders["application_status"].classes_)
#     st.pyplot(fig)

# elif task == "⚠️ Default Flag Prediction":
#     st.subheader("Default Flag Prediction")
#     X = df.drop(columns=["customer_id", "application_status", "default_flag", "dpd_bucket", "total_delay"])
#     y = df["default_flag"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     st.text("Classification Report:")
#     st.text(classification_report(y_test, y_pred))

# elif task == "⏳ Processing Days Prediction (Regression)":
#     st.subheader("Processing Days Regression")
#     X = df.drop(columns=["customer_id", "application_status", "default_flag", "dpd_bucket", "processing_days", "total_delay"])
#     y = df["processing_days"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     mae = mean_absolute_error(y_test, y_pred)
#     st.text(f"Mean Absolute Error: {mae:.2f} days")

#     fig, ax = plt.subplots()
#     ax.scatter(y_test, y_pred, alpha=0.5)
#     ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
#     ax.set_xlabel("Actual")
#     ax.set_ylabel("Predicted")
#     ax.set_title("Processing Time Prediction")
#     st.pyplot(fig)

# elif task == "📦 Delay Bucket Classification":
#     st.subheader("DPD Bucket Classification")
#     X = df.drop(columns=["customer_id", "application_status", "default_flag", "dpd_bucket", "processing_days", "total_delay"])
#     y = df["dpd_bucket"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     st.text("Classification Report:")
#     st.text(classification_report(y_test, y_pred))

# elif task == "🚨 Anomaly Detection":
#     st.subheader("Anomaly Detection")
#     num_cols = [
#         "age", "salary", "credit_score", "document_completeness",
#         "past_payment_delays", "credit_utilization", "processing_days", "document_submission_delay"
#     ]
#     iso = IsolationForest(contamination=0.05, random_state=42)
#     iso.fit(df[num_cols])
#     df["anomaly_flag"] = iso.predict(df[num_cols])

#     fig, ax = plt.subplots()
#     sns.countplot(x="anomaly_flag", data=df, ax=ax)
#     ax.set_title("Anomaly Detection (1 = Normal, -1 = Suspicious)")
#     st.pyplot(fig)

#     st.success(f"Suspicious Applications Detected: {(df['anomaly_flag'] == -1).sum()}")







import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import classification_report, mean_absolute_error, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit App Analyzer", layout="wide")
st.title("📊 Credit Card Application ML Analyzer")

# -------------- Sample CSV --------------
sample_csv = """
customer_id,age,salary,credit_score,employment_type,application_source,document_completeness,past_payment_delays,credit_utilization,document_submission_delay,processing_days,application_status,default_flag,latitude,longitude
C001,35,60000,700,Salaried,Web,0.9,2,0.4,5,10,Approved,0,40.7128,-74.0060
C002,42,80000,720,Self-Employed,Agent,0.8,0,0.3,2,5,Approved,0,34.0522,-118.2437
C003,28,40000,650,Salaried,Web,0.7,5,0.5,10,15,Rejected,1,41.8781,-87.6298
C004,50,95000,800,Salaried,Branch,1.0,1,0.2,1,3,Approved,0,29.7604,-95.3698
C005,38,50000,620,Self-Employed,Agent,0.6,3,0.6,7,12,Pending,1,39.9526,-75.1652
"""

# -------------- Bucket Function --------------
def assign_bucket(days):
    if days <= 30: return '0-30'
    elif days <= 60: return '30-60'
    elif days <= 90: return '60-90'
    elif days <= 120: return '90-120'
    elif days <= 150: return '120-150'
    elif days <= 180: return '150-180'
    else: return '180+'

# -------------- Step 1: Load Data --------------
st.sidebar.header("Step 1: Choose Data Source")
data_source = st.sidebar.radio("Source", ["Upload CSV", "Use Sample Data"])

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your credit_card_application_data.csv", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Uploaded CSV loaded successfully.")
    else:
        st.warning("⚠️ Please upload a CSV file to proceed.")
        st.stop()
else:
    df = pd.read_csv(StringIO(sample_csv))
    st.success("✅ Using built-in sample CSV data.")

# -------------- Step 2: Preprocess Data --------------
if "delay_bucket" in df.columns:
    df.drop(columns=["delay_bucket"], inplace=True)

df["total_delay"] = df["processing_days"] + df["document_submission_delay"]
df["dpd_bucket"] = df["total_delay"].apply(assign_bucket)

label_cols = ["employment_type", "application_source", "application_status", "dpd_bucket"]
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

if st.checkbox("🔍 Show Processed DataFrame"):
    st.dataframe(df)

# -------------- Map Option --------------
st.sidebar.header("Step 3: Map Visualization")
show_map = st.sidebar.checkbox("Show Customer Location Map")

if show_map:
    if "latitude" in df.columns and "longitude" in df.columns:
        st.subheader("🗺️ Customer Locations Map")
        st.map(df[["latitude", "longitude"]].dropna())
    else:
        st.warning("⚠️ No latitude and longitude columns found for map visualization.")

# -------------- Data Visualization --------------
st.sidebar.header("Step 4: Data Visualization")

plot_type = st.sidebar.selectbox("Choose Plot Type", [
    "Select",
    "Histogram",
    "Boxplot",
    "Correlation Heatmap",
    "Countplot",
    "Scatter Plot"
])

if plot_type == "Histogram":
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    col = st.sidebar.selectbox("Select numeric column", numeric_cols)
    bins = st.sidebar.slider("Number of bins", 5, 100, 20)
    st.subheader(f"Histogram of {col}")
    fig, ax = plt.subplots()
    ax.hist(df[col].dropna(), bins=bins, color='skyblue', edgecolor='black')
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

elif plot_type == "Boxplot":
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['int', 'category']).columns.tolist()
    # Filter only encoded categorical columns (low cardinality)
    cat_cols = [c for c in cat_cols if df[c].nunique() < 20]
    if not cat_cols:
        st.warning("No suitable categorical column available for boxplot grouping.")
    else:
        y_col = st.sidebar.selectbox("Select numeric column (Y)", numeric_cols)
        x_col = st.sidebar.selectbox("Select categorical column (X)", cat_cols)
        st.subheader(f"Boxplot of {y_col} by {x_col}")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[x_col], y=df[y_col], ax=ax)
        st.pyplot(fig)

elif plot_type == "Correlation Heatmap":
    st.subheader("Correlation Heatmap of Numeric Features")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif plot_type == "Countplot":
    cat_cols = df.select_dtypes(include=['int', 'category']).columns.tolist()
    # Use only low cardinality categorical
    cat_cols = [c for c in cat_cols if df[c].nunique() < 30]
    if not cat_cols:
        st.warning("No categorical column found for countplot.")
    else:
        col = st.sidebar.selectbox("Select categorical column", cat_cols)
        st.subheader(f"Countplot of {col}")
        fig, ax = plt.subplots()
        sns.countplot(x=df[col], ax=ax)
        ax.set_xlabel(col)
        st.pyplot(fig)

elif plot_type == "Scatter Plot":
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Need at least two numeric columns for scatter plot.")
    else:
        x_col = st.sidebar.selectbox("X axis", numeric_cols)
        y_col = st.sidebar.selectbox("Y axis", numeric_cols, index=1)
        color_col = st.sidebar.selectbox("Color by (optional)", ["None"] + df.columns.tolist())
        st.subheader(f"Scatter plot: {y_col} vs {x_col}")
        fig, ax = plt.subplots()
        if color_col != "None":
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=color_col, ax=ax)
        else:
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
        st.pyplot(fig)

# -------------- Step 5: Choose ML Task --------------
st.sidebar.header("Step 5: Choose ML Task")
task = st.sidebar.selectbox("Functionality", [
    "Select",
    "📌 Application Status Classification",
    "⚠️ Default Flag Prediction",
    "⏳ Processing Days Prediction (Regression)",
    "📦 Delay Bucket Classification",
    "🚨 Anomaly Detection"
])

# -------------- Step 6: Run Models --------------
if task == "📌 Application Status Classification":
    st.subheader("📌 Application Status Classification")
    X = df.drop(columns=["customer_id", "application_status", "default_flag", "dpd_bucket", "total_delay"])
    y = df["application_status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                xticklabels=label_encoders["application_status"].classes_,
                yticklabels=label_encoders["application_status"].classes_)
    st.pyplot(fig)

elif task == "⚠️ Default Flag Prediction":
    st.subheader("⚠️ Default Flag Classification")
    X = df.drop(columns=["customer_id", "application_status", "default_flag", "dpd_bucket", "total_delay"])
    y = df["default_flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

elif task == "⏳ Processing Days Prediction (Regression)":
    st.subheader("⏳ Processing Days Regression")
    X = df.drop(columns=["customer_id", "application_status", "default_flag", "dpd_bucket", "processing_days", "total_delay"])
    y = df["processing_days"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st.text(f"📉 Mean Absolute Error: {mae:.2f} days")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Processing Time Prediction")
    st.pyplot(fig)

elif task == "📦 Delay Bucket Classification":
    st.subheader("📦 DPD Bucket Classification")
    X = df.drop(columns=["customer_id", "application_status", "default_flag", "dpd_bucket", "processing_days", "total_delay"])
    y = df["dpd_bucket"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

elif task == "🚨 Anomaly Detection":
    st.subheader("🚨 Anomaly Detection using Isolation Forest")
    num_cols = [
        "age", "salary", "credit_score", "document_completeness",
        "past_payment_delays", "credit_utilization", "processing_days", "document_submission_delay"
    ]
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(df[num_cols])
    df["anomaly_flag"] = iso.predict(df[num_cols])

    fig, ax = plt.subplots()
    sns.countplot(x="anomaly_flag", data=df, ax=ax)
    ax.set_title("Anomaly Detection (1 = Normal, -1 = Suspicious)")
    st.pyplot(fig)

    st.success(f"🚨 Suspicious Applications Detected: {(df['anomaly_flag'] == -1).sum()}")

