# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Configure app
st.set_page_config(page_title="EduTrack: Student Learning Gap Analyzer", layout="wide")

# Sidebar: About Section
with st.sidebar:
    st.sidebar.title("📘 About ")
    st.sidebar.markdown("""
EduTrack is an interactive **AI-powered web app** designed to analyze student performance data, 
identify learning gaps, and generate actionable insights for **students**, **teachers**, and **policymakers**.

✅ Uses: Rule-based + Machine Learning (Random Forest) models  
✅ Visualizes: Heatmaps, bar charts, pie charts  
✅ Supports: Downloadable reports  
✅ Future: Integration of real-world datasets, geospatial analysis, dashboard improvements
""")

# Sidebar: App Controls
st.sidebar.title("⚙️ App Controls")
show_model = st.sidebar.checkbox("Show Model Training Results", value=True)
show_visuals = st.sidebar.checkbox("Show Visualizations", value=True)
show_downloads = st.sidebar.checkbox("Show Download Options", value=True)
specific_lo = st.sidebar.selectbox("Select Specific LO for Pie Chart", sorted([f'8.M.LO{i}' for i in range(6, 20)]), index=3)
top_n_schools = st.sidebar.slider("Top N Schools (Heatmap/Pie)", 3, 15, 10)

# Sidebar: Team Info
st.sidebar.title("👥 Team & Contact")
st.sidebar.markdown("""
Created by **DeepSpatial**  
📧 Contact: roshanraju7654@gmail.com  
🌐 [Project GitHub](https://github.com/yourproject)  
""")

# Main Title
st.title("📊 EduTrack: Student Learning Gap Analyzer")

# STEP 1: File Upload
uploaded_file = st.file_uploader("📁 Upload your student dataset CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset loaded! Shape: {df.shape}")
    st.dataframe(df.head())

    st.sidebar.markdown("### 📊 Quick Summary")
    st.sidebar.metric("Number of Students", df['Student_ID'].nunique())
    st.sidebar.metric("Number of Schools", df['School_Name'].nunique())

    # Prerequisites mapping
    prerequisites = {
        '8.M.LO9': ['7.M.LO9', '6.M.LO8'],
        '8.M.LO7': ['7.M.LO6', '6.M.LO7'],
        '8.M.LO17': ['7.M.LO17', '6.M.LO17'],
        '8.M.LO18': ['7.M.LO18', '6.M.LO16'],
        '8.M.LO6': ['7.M.LO11', '6.M.LO7'],
        '8.M.LO10': ['7.M.LO10', '6.M.LO9']
    }

    # Learning gap function
    def get_learning_gaps(student_id, df, prerequisites):
        student_data = df[df['Student_ID'] == student_id]
        weak_los = set()
        for col in df.columns:
            if col.endswith('_Score'):
                score_value = student_data[col].values[0]
                if pd.isnull(score_value): continue
                if score_value == 0:
                    qid = col.split('_')[0]
                    lo_col = f"{qid}_LO_Concept"
                    if lo_col in student_data:
                        lo_code = student_data[lo_col].values[0].split(' - ')[0]
                        weak_los.add(lo_code)
        review_needed = set()
        for lo in weak_los:
            review_needed.update(prerequisites.get(lo, []))
        return {'Weak_LOs': list(weak_los), 'Review_Recommended': list(review_needed)}

    # Prepare ML data
    def prepare_ml_data(df, prerequisites):
        score_cols = [col for col in df.columns if col.endswith('_Score')]
        X = df[score_cols].copy()
        y = []
        for _, row in df.iterrows():
            gaps = get_learning_gaps(row['Student_ID'], df, prerequisites)
            labels = [1 if f'8.M.LO{i}' in gaps['Weak_LOs'] else 0 for i in range(6, 20)]
            y.append(labels)
        y = pd.DataFrame(y, columns=[f'8.M.LO{i}' for i in range(6, 20)])
        return X, y

    # Train model
    @st.cache_data
    def train_model(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return clf, y_test, y_pred

    # Analyze button
    if st.button("🚀 Analyze Learning Gaps and Train Model"):
        with st.spinner("⏳ Processing... Please wait."):
            X, y = prepare_ml_data(df, prerequisites)
            model, y_test, y_pred = train_model(X, y)
            st.success("🎯 Model training completed!")

            if show_model:
                st.subheader("📄 Classification Report (First LO: 8.M.LO6)")
                report = classification_report(y_test.iloc[:, 0], y_pred[:, 0], target_names=["Strong", "Weak"], output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

            # Analyze gaps
            gap_records = []
            for sid in df['Student_ID'].unique():
                gaps = get_learning_gaps(sid, df, prerequisites)
                gap_records.append({
                    'Student_ID': sid,
                    'Weak_LOs': gaps['Weak_LOs'],
                    'Review_Recommended': gaps['Review_Recommended']
                })
            gap_df = pd.DataFrame(gap_records)
            school_meta = df[['Student_ID', 'School_Name', 'School_Code']].drop_duplicates()
            gap_df = gap_df.merge(school_meta, on='Student_ID', how='left')

            school_summary = (
                gap_df.explode('Weak_LOs')
                      .groupby('School_Name')['Weak_LOs']
                      .value_counts()
                      .unstack(fill_value=0)
                      .sort_index(axis=1)
            )
            district_summary = gap_df.explode('Weak_LOs')['Weak_LOs'].value_counts()

            if show_visuals:
                st.subheader("🔥 Visualizations")

                # Heatmap
                school_total_weak = school_summary.sum(axis=1).sort_values(ascending=False)
                top_schools = school_total_weak.head(top_n_schools).index
                school_summary_top = school_summary.loc[top_schools]
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                sns.heatmap(school_summary_top, cmap='OrRd', annot=True, fmt='d', linewidths=0.5, ax=ax1)
                ax1.set_title(f'Top {top_n_schools} Schools: Weak Learning Outcomes Heatmap')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig1)

                # Barplot
                top_weak_los = district_summary.sort_values(ascending=False).head(5)
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                sns.barplot(x=top_weak_los.index, y=top_weak_los.values, palette='Blues_r', ax=ax2)
                ax2.set_title('Top 5 Weak Learning Outcomes (District)')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig2)

                # Pie chart
                lo_counts_per_school = (
                    gap_df.explode('Weak_LOs')
                          .query('Weak_LOs == @specific_lo')
                          .groupby('School_Name')
                          .size()
                          .sort_values(ascending=False)
                )
                top_lo_schools = lo_counts_per_school.head(top_n_schools)
                if not top_lo_schools.empty:
                    fig3, ax3 = plt.subplots(figsize=(8, 8))
                    ax3.pie(top_lo_schools, labels=top_lo_schools.index, autopct='%1.1f%%', startangle=140)
                    ax3.set_title(f'Top {top_n_schools} Schools with Weak Students in {specific_lo}')
                    st.pyplot(fig3)
                else:
                    st.warning(f"No weak students found for {specific_lo}.")

            if show_downloads:
                st.subheader("💾 Download Reports")
                st.download_button("Download Classification Report CSV", report_df.to_csv().encode(), "classification_report.csv", "text/csv")
                st.download_button("Download School Summary CSV", school_summary.to_csv().encode(), "school_summary.csv", "text/csv")
                st.download_button("Download District Summary CSV", district_summary.to_csv().encode(), "district_summary.csv", "text/csv")

else:
    st.info("👆 Please upload a dataset CSV file to begin.")

# Footer
st.markdown(
    "<div style='text-align: center;'><strong>EduTrack</strong> || DeepSpatial</div>", 
    unsafe_allow_html=True
)
