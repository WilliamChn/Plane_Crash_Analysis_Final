import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

import streamlit as st

# Load your existing dataset (same logic as in your notebook)
file_url = "https://raw.githubusercontent.com/Suh716/Airplane-Crash-Data-Analysis-and-EDAs/refs/heads/dev/Airplane_Crashes_and_Fatalities_Since_1908.csv"
df = pd.read_csv(file_url)

#show entire dataset
combined_df = df.copy()

# Load saved user entries if they exist
try:
    user_data = pd.read_csv("user_entries.csv")
    combined_df = pd.concat([combined_df, user_data], ignore_index=True)
except FileNotFoundError:
    user_data = pd.DataFrame(columns=df.columns)

st.title("Airplane Crash Data Viewer & Input")

# Display the current combined dataframe
st.subheader("Current Combined Data")
st.dataframe(combined_df)

# Input form
st.subheader("Add Your Own Row")
with st.form("user_input_form"):
    user_input = {}
    for col in df.columns:
        user_input[col] = st.text_input(col)
    submitted = st.form_submit_button("Submit")

# On submission: add to user_data and save
if submitted:
    new_row = pd.DataFrame([user_input])
    user_data = pd.concat([user_data, new_row], ignore_index=True)
    user_data.to_csv("user_entries.csv", index=False)
    st.success("Row added successfully. Refresh the page to see updated table.")

import matplotlib.pyplot as plt

st.subheader("Airplane Crashes Over Time")

# Make sure 'Date' is in datetime format
combined_df["Date"] = pd.to_datetime(combined_df["Date"], errors="coerce")

# Extract year
combined_df["Year"] = combined_df["Date"].dt.year

# Count crashes per year
crash_trend = combined_df.groupby("Year").size()

# Plot the trend using matplotlib
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(crash_trend, marker='o', linestyle='-')
ax.set_xlabel("Year")
ax.set_ylabel("Number of Crashes")
ax.set_title("Airplane Crashes Over Time")
ax.grid(True)

# Display in Streamlit
st.pyplot(fig)


# Ensure datetime conversion
combined_df["Date"] = pd.to_datetime(combined_df["Date"], errors="coerce")

#  Crashes by Month and Day of the Week
st.subheader("Crashes by Month and Day of the Week")

combined_df["Month"] = combined_df["Date"].dt.month
combined_df["Day_of_Week"] = combined_df["Date"].dt.dayofweek  # 0 = Monday, 6 = Sunday

# Crashes by Month
fig_month, ax_month = plt.subplots(figsize=(12, 5))
combined_df["Month"].value_counts().sort_index().plot(kind='bar', color='skyblue', ax=ax_month)
ax_month.set_xlabel("Month")
ax_month.set_ylabel("Number of Crashes")
ax_month.set_title("Crashes by Month")
st.pyplot(fig_month)

# Crashes by Day of Week
fig_day, ax_day = plt.subplots(figsize=(12, 5))
combined_df["Day_of_Week"].value_counts().sort_index().plot(kind='bar', color='lightcoral', ax=ax_day)
ax_day.set_xlabel("Day of the Week (0 = Monday, 6 = Sunday)")
ax_day.set_ylabel("Number of Crashes")
ax_day.set_title("Crashes by Day of the Week")
st.pyplot(fig_day)


# Airlines with Most Crashes
st.subheader("Top 15 Airlines with Most Crashes")

df_cleaned = combined_df[combined_df['Operator'].notna() & (combined_df['Operator'].str.strip() != '')]
top_operators = df_cleaned['Operator'].value_counts().head(15)

fig_ops, ax_ops = plt.subplots(figsize=(20, 5))
top_operators.plot(kind='bar', color='purple', ax=ax_ops)
ax_ops.set_xlabel("Operator")
ax_ops.set_ylabel("Number of Crashes")
ax_ops.set_title("Top 15 Airlines with Most Crashes")
ax_ops.tick_params(axis='x', rotation=45)
st.pyplot(fig_ops)

# Aircraft Model Analysis

st.subheader("Top 10 Aircraft Models with Most Crashes")

top_aircrafts = combined_df['Type'].value_counts().head(10)

fig_type, ax_type = plt.subplots(figsize=(18, 5))
top_aircrafts.plot(kind='bar', color='green', ax=ax_type)
ax_type.set_xlabel("Aircraft Type")
ax_type.set_ylabel("Number of Crashes")
ax_type.set_title("Top 10 Aircraft Models with Most Crashes")
ax_type.tick_params(axis='x', rotation=45)
st.pyplot(fig_type)

# Distribution of Fatalities per Crash

st.subheader("Distribution of Fatalities per Airplane Crash")

df_fatalities = combined_df['Fatalities'].dropna()

fig_fatal, ax_fatal = plt.subplots(figsize=(10, 5))
ax_fatal.hist(df_fatalities, bins=30, color='red', edgecolor='black', alpha=0.7)
ax_fatal.set_xlabel("Number of Fatalities per Crash")
ax_fatal.set_ylabel("Frequency")
ax_fatal.set_title("Distribution of Fatalities per Airplane Crash")
ax_fatal.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig_fatal)


#  Passenger Load vs. Fatality Rate

st.subheader("Passenger Load vs. Fatality Rate in Airplane Crashes")

df_filtered = combined_df[['Aboard', 'Fatalities']].dropna()
df_filtered = df_filtered[df_filtered['Aboard'] > 0]  # Avoid division by zero
df_filtered['Fatality Rate'] = df_filtered['Fatalities'] / df_filtered['Aboard']

fig_scatter, ax_scatter = plt.subplots(figsize=(10, 5))
ax_scatter.scatter(df_filtered['Aboard'], df_filtered['Fatality Rate'], alpha=0.5, color='blue')
ax_scatter.set_xlabel("Number of People Aboard")
ax_scatter.set_ylabel("Fatality Rate")
ax_scatter.set_title("Passenger Load vs. Fatality Rate in Airplane Crashes")
ax_scatter.axhline(y=1, color='r', linestyle='--', label='100% Fatalities')
ax_scatter.legend()
ax_scatter.grid()
st.pyplot(fig_scatter)

# Word Cloud from Crash Summaries

st.subheader("Common Keywords in Crash Summaries")

summary_text = " ".join(combined_df['Summary'].dropna()).lower()
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(summary_text)

fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis("off")
ax_wc.set_title("Common Keywords in Crash Summaries")
st.pyplot(fig_wc)





#ML algorithms
st.subheader("ML and statistical algorithms")

st.subheader("Linear Regression: Predicting Fatalities")

# Clean and prep data
df_clean = combined_df[['Aboard', 'Ground', 'Fatalities', 'Date']].dropna()
df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
df_clean['Year'] = df_clean['Date'].dt.year
df_clean = df_clean.drop(columns=['Date'])

X_reg = df_clean[['Aboard', 'Ground', 'Year']]
y_reg = df_clean['Fatalities']

# Split and scale
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reg)
X_test_scaled = scaler.transform(X_test_reg)

# Train and predict
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train_reg)
y_pred = lin_reg.predict(X_test_scaled)

# Evaluation
mae = mean_absolute_error(y_test_reg, y_pred)
mse = mean_squared_error(y_test_reg, y_pred)
rmse = np.sqrt(mse)

# Plot
fig_lr, ax_lr = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=y_test_reg, y=y_pred, alpha=0.6, ax=ax_lr)
ax_lr.set_xlabel("Actual Fatalities")
ax_lr.set_ylabel("Predicted Fatalities")
ax_lr.set_title("Linear Regression: Actual vs Predicted Fatalities")
st.pyplot(fig_lr)

# Metrics display
st.markdown("**Linear Regression Metrics:**")
st.markdown(f"- MAE: `{mae:.2f}`")
st.markdown(f"- MSE: `{mse:.2f}`")
st.markdown(f"- RMSE: `{rmse:.2f}`")


import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, f1_score, confusion_matrix, classification_report

st.subheader("Logistic Regression: Predicting High Fatality Crashes")

# Start timing
start_time = time.time()

# Create High_Fatality target
df_clean['High_Fatality'] = (df_clean['Fatalities'] > 50).astype(int)

# Feature and label
X_cls = df_clean[['Aboard', 'Ground', 'Year']]
y_cls = df_clean['High_Fatality']

# Train-test split
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_cls_scaled = scaler.fit_transform(X_train_cls)
X_test_cls_scaled = scaler.transform(X_test_cls)

# Train Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_cls_scaled, y_train_cls)

# Predict
y_pred_cls = log_reg.predict(X_test_cls_scaled)

#  End timing
end_time = time.time()
exec_time = round(end_time - start_time, 2)

# Metrics
accuracy = accuracy_score(y_test_cls, y_pred_cls)
precision = precision_score(y_test_cls, y_pred_cls)
f1 = f1_score(y_test_cls, y_pred_cls)
conf_matrix = confusion_matrix(y_test_cls, y_pred_cls)
class_report_text = classification_report(y_test_cls, y_pred_cls)

#  Confusion Matrix Plot
fig_conf, ax_conf = plt.subplots(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Fatality', 'High Fatality'],
            yticklabels=['Low Fatality', 'High Fatality'],
            ax=ax_conf)
ax_conf.set_xlabel("Predicted Label")
ax_conf.set_ylabel("True Label")
ax_conf.set_title("Logistic Regression: Confusion Matrix")
st.pyplot(fig_conf)

#  Display Metrics on Streamlit
st.markdown("### Logistic Regression Metrics")
st.write(f"Execution Time: `{exec_time} seconds`")
st.write(f"Accuracy: `{accuracy:.2f}`")
st.write(f"Precision: `{precision:.2f}`")
st.write(f"F1 Score: `{f1:.2f}`")

st.markdown("###  Classification Report")
st.text(class_report_text)



# Naive


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.subheader("Naive Bayes: Predicting Crash Severity from Summary")

# Define severity categories
def classify_severity(fatalities):
    if fatalities == 0:
        return "No Fatalities"
    elif fatalities < 50:
        return "Low"
    elif fatalities < 150:
        return "Medium"
    else:
        return "High"

df['Severity'] = df['Fatalities'].apply(classify_severity)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['Summary'].fillna(""), df['Severity'], test_size=0.3, random_state=42)

# Naive Bayes pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english', max_features=5000)),
    ('nb', MultinomialNB())
])

# Train the model
clf.fit(X_train, y_train)

# Accuracy
accuracy = clf.score(X_test, y_test)

# Feature importance extraction
vectorizer = clf.named_steps['vectorizer']
feature_names = np.array(vectorizer.get_feature_names_out())
nb_model = clf.named_steps['nb']
top_features = np.argsort(nb_model.feature_log_prob_[0])[-15:]

# Plot top words
fig_nb, ax_nb = plt.subplots(figsize=(8, 6))
ax_nb.barh(feature_names[top_features], nb_model.feature_log_prob_[0][top_features], color="blue")
ax_nb.set_xlabel("Log Probability")
ax_nb.set_ylabel("Words")
ax_nb.set_title("Impact of Words on Predictions (Multinomial Naive Bayes)")
st.pyplot(fig_nb)

# Display accuracy
st.markdown(f"**Naive Bayes Model Accuracy:** `{accuracy * 100:.2f}%`")


#3 random forest

from sklearn.ensemble import RandomForestClassifier

st.subheader("Random Forest: Crash Severity Classification")

# Define crash severity if not already defined
def classify_severity(fatalities):
    if fatalities == 0:
        return "No Fatalities"
    elif fatalities < 50:
        return "Low"
    elif fatalities < 150:
        return "Medium"
    else:
        return "High"

df['Severity'] = df['Fatalities'].apply(classify_severity)

# Encode categorical features
label_encoders = {}
for col in ['Operator', 'Type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define features and target
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year


X = df[['Year', 'Aboard', 'Operator', 'Type']]
y = df['Severity']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred_rf = rf_model.predict(X_test)

# Feature importance
feature_importance = rf_model.feature_importances_
features = X.columns

# Plot feature importance
fig_rf, ax_rf = plt.subplots(figsize=(8, 6))
ax_rf.barh(features, feature_importance, color="blue")
ax_rf.set_xlabel("Importance Score")
ax_rf.set_title("Feature Importance in Predicting Crash Severity (Random Forest)")
st.pyplot(fig_rf)

# Show accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
st.markdown(f"**Random Forest Model Accuracy:** `{accuracy_rf * 100:.2f}%`")