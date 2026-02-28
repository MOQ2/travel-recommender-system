"""
Fix script for malformed source_file entries in the cleaned dataset.
This directly fixes the source_file column to match the expected pattern.
"""
import pandas as pd
import re

# Load the dataset
df = pd.read_csv('data/processed/cleaned_dataset.csv')

# Remove the student_id_fixed column if it exists from previous runs
if 'student_id_fixed' in df.columns:
    df = df.drop(columns=['student_id_fixed'])

print(f"Original dataset: {len(df)} rows")

def fix_source_file(source_file):
    """
    Fix malformed source_file names to match the expected pattern: prefix-STUDENTID.csv
    """
    original = source_file
    
    # Already matches the expected pattern
    if re.search(r'-\d{7}\.csv$', source_file):
        return source_file
    
    # Extract the prefix (first part before hyphen)
    prefix_match = re.match(r'^(\d+)-', source_file)
    prefix = prefix_match.group(1) if prefix_match else None
    
    # Try to find a 7-digit student ID anywhere in the filename
    id_match = re.search(r'(\d{7})', source_file)
    if id_match and prefix:
        student_id = id_match.group(1)
        fixed = f"{prefix}-{student_id}.csv"
        return fixed
    
    # If we only have a prefix (e.g., '2938327-StudentID.csv'), use the prefix as ID
    if prefix and len(prefix) == 7:
        return f"{prefix}-{prefix}.csv"
    
    # Fallback: return original (will result in NaN, handled later)
    return source_file

# Show problematic files before fix
problematic_mask = ~df['source_file'].str.match(r'.*-\d{7}\.csv$')
problematic_count = problematic_mask.sum()
print(f"\nProblematic source_file entries: {problematic_count}")

if problematic_count > 0:
    print("\nBefore fix:")
    for sf in df[problematic_mask]['source_file'].unique():
        print(f"  '{sf}'")

# Apply the fix
df['source_file'] = df['source_file'].apply(fix_source_file)

# Verify fix worked
still_problematic_mask = ~df['source_file'].str.match(r'.*-\d{7}\.csv$')
still_problematic = still_problematic_mask.sum()

print(f"\nAfter fix:")
print(f"  Fixed: {problematic_count - still_problematic} entries")
print(f"  Still problematic: {still_problematic} entries")

if still_problematic > 0:
    print("\nStill cannot fix:")
    for sf in df[still_problematic_mask]['source_file'].unique():
        print(f"  '{sf}'")
    # Remove unfixable rows
    df = df[~still_problematic_mask]
    print(f"\nRemoved {still_problematic} unfixable rows")

# Verify extraction works now
df_test = df.copy()
df_test['student_id'] = df_test['source_file'].str.extract(r'-(\d+)\.csv$')[0]
missing_ids = df_test['student_id'].isna().sum()
unique_students = df_test['student_id'].nunique()

print(f"\nVerification:")
print(f"  Total rows: {len(df)}")
print(f"  Missing student_id after extraction: {missing_ids}")
print(f"  Unique students: {unique_students}")

# Save the fixed dataset
df.to_csv('data/processed/cleaned_dataset.csv', index=False)
print(f"\nSaved fixed dataset: {len(df)} rows")
print("Done! The EDA notebook should now work without NaN errors.")
