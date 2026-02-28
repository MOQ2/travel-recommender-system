
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency
import sys
import os

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eda_utils import add_derived_columns

def cramers_v(confusion_matrix):
    """Calculate Cramér's V statistic for categorical correlation"""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

def main():
    print("=" * 80)
    print("RUNNING EDA VERIFICATION")
    print("=" * 80)

    # 1. Load Data
    data_path = Path('../data/processed/cleaned_dataset.csv')
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    df['student_id'] = df['source_file'].str.extract(r'-(\d+)\.csv$')[0]
    
    print(f"Loaded {len(df)} records.")
    print("Deriving missing columns...")
    
    # 2. Add Derived Columns
    df = add_derived_columns(df)
    
    # Save the enriched dataset for use in notebooks
    output_path = Path('../data/processed/cleaned_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"✓ Saved enriched dataset to {output_path}")
    
    print(f"Columns after derivation: {list(df.columns)}")
    
    if 'Activity_Type' not in df.columns:
        print("Error: Activity_Type derivation failed.")
        return

    # 3. Statistics Analysis (Replicating eda_statistics.ipynb logic)
    print("\n--- Statistics Analysis ---")
    print("Top Activities:")
    print(df['Activity_Type'].value_counts().head())
    
    print("\nTop Weather Types:")
    print(df['Weather_Type'].value_counts().head())

    # 4. Correlation Analysis (Replicating eda_correlation_analysis.ipynb logic)
    print("\n--- Correlation Analysis ---")
    cat_vars = ['Weather_Type', 'Activity_Type', 'Mood_Category']
    correlation_matrix = np.zeros((len(cat_vars), len(cat_vars)))

    for i, var1 in enumerate(cat_vars):
        for j, var2 in enumerate(cat_vars):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                confusion = pd.crosstab(df[var1], df[var2])
                correlation_matrix[i, j] = cramers_v(confusion.values)

    corr_df = pd.DataFrame(correlation_matrix, index=cat_vars, columns=cat_vars)
    print("\nCramér's V Correlation Matrix:")
    print(corr_df.round(3))

    # 5. Visual Analysis (Replicating eda_visual_analysis.ipynb logic)
    print("\n--- Visual Analysis --")
    # Text length analysis
    df['description_length'] = df['Description'].fillna('').astype(str).apply(len)
    df['description_word_count'] = df['Description'].fillna('').astype(str).apply(lambda x: len(x.split()))
    
    print(f"Avg Description Length: {df['description_length'].mean():.2f}")
    
    # Check for plotting issues
    try:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Activity_Type', y='description_word_count', data=df)
        plt.title('Description Word Count by Activity')
        plt.xticks(rotation=45)
        # plt.show() # Don't show in CLI
        print("Plot generation successful.")
        plt.close()
    except Exception as e:
        print(f"Plot generation failed: {e}")

    # 6. Student Analysis (Replicating eda_student_analysis.ipynb logic)
    print("\n--- Student Analysis ---")
    student_features = []
    
    unique_students = df['student_id'].unique()
    print(f"Processing {len(unique_students)} students...")

    for student_id in unique_students:
        student_df = df[df['student_id'] == student_id]
        
        # Basic counts
        n_places = len(student_df)
        
        # Category diversity (unique counts)
        n_activities = student_df['Activity_Type'].nunique()
        n_weather = student_df['Weather_Type'].nunique()
        n_moods = student_df['Mood_Category'].nunique()
        n_countries = student_df['Country_Standardized'].nunique()
        
        # Most common preferences
        primary_activity = student_df['Activity_Type'].mode()[0] if len(student_df['Activity_Type'].mode()) > 0 else 'Unknown'
        
        student_features.append({
            'student_id': student_id,
            'n_places': n_places,
            'n_activities': n_activities,
            'n_weather': n_weather,
            'n_moods': n_moods,
            'n_countries': n_countries,
            'primary_activity': primary_activity
        })
    
    student_features_df = pd.DataFrame(student_features)
    print(f"Created features for {len(student_features_df)} students.")
    print("Average activities per student: {:.2f}".format(student_features_df['n_activities'].mean()))
    print("Average countries per student: {:.2f}".format(student_features_df['n_countries'].mean()))

    print("\n✅ EDA Verification Completed Successfully.")

if __name__ == "__main__":
    main()
