import pandas as pd

df = pd.read_csv('data/processed/cleaned_dataset.csv')
print('Dataset shape:', df.shape)
print('\n=== Missing Values ===')
print(df.isnull().sum().to_string())
print('\n=== Value Counts for Key Columns ===')
for col in ['Activity_Type', 'Weather_Type', 'Mood_Category', 'Country_Standardized']:
    print(f'\n{col}:')
    print(f'  Unique values: {df[col].nunique()}')
    print(f'  NaN count: {df[col].isnull().sum()}')
    if df[col].isnull().sum() > 0:
        print(f'  WARNING: Contains missing values!')

# Check source_file pattern extraction
df['student_id'] = df['source_file'].str.extract(r'-(\d+)\.csv$')[0]
print('\n=== Student ID Extraction ===')
print(f'Total rows: {len(df)}')
print(f'Rows with valid student_id: {df["student_id"].notna().sum()}')
print(f'Rows with NaN student_id: {df["student_id"].isna().sum()}')
if df['student_id'].isna().sum() > 0:
    print('\nRows with missing student_id:')
    print(df[df['student_id'].isna()]['source_file'].unique())
