"""Quick analysis script to understand data issues"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Paths
data_dir = Path('data')

# Load data
df = pd.read_csv(data_dir / 'processed' / 'cleaned_dataset.csv')
train_pairs = pd.read_csv(data_dir / 'splits' / 'train_pairs.csv')
val_pairs = pd.read_csv(data_dir / 'splits' / 'val_pairs.csv')

print("="*60)
print("DATA ANALYSIS")
print("="*60)

print(f"\n1. Dataset: {len(df)} places")
print(f"   Columns: {df.columns.tolist()}")

print(f"\n2. Train pairs: {len(train_pairs)}")
print(f"   Val pairs: {len(val_pairs)}")
print(f"   Train students: {train_pairs['student_id'].nunique()}")
print(f"   Val students: {val_pairs['student_id'].nunique()}")

train_students = set(train_pairs['student_id'].unique())
val_students = set(val_pairs['student_id'].unique())
overlap = train_students.intersection(val_students)
print(f"   Student overlap: {len(overlap)} ({100*len(overlap)/len(val_students):.1f}%)")

print(f"\n3. Label distribution:")
print(f"   Train - Pos: {(train_pairs['label']==1).sum()}, Neg: {(train_pairs['label']==0).sum()}")
print(f"   Val - Pos: {(val_pairs['label']==1).sum()}, Neg: {(val_pairs['label']==0).sum()}")
print(f"   Positive ratio: {(train_pairs['label']==1).mean():.2%}")

print(f"\n4. Index ranges:")
print(f"   Train anchor_idx: {train_pairs['anchor_idx'].min()} - {train_pairs['anchor_idx'].max()}")
print(f"   Train pair_idx: {train_pairs['pair_idx'].min()} - {train_pairs['pair_idx'].max()}")
print(f"   Val anchor_idx: {val_pairs['anchor_idx'].min()} - {val_pairs['anchor_idx'].max()}")
print(f"   Val pair_idx: {val_pairs['pair_idx'].min()} - {val_pairs['pair_idx'].max()}")
print(f"   Dataset size: {len(df)}")

# Check if indices are valid
max_idx = max(train_pairs['anchor_idx'].max(), train_pairs['pair_idx'].max(),
              val_pairs['anchor_idx'].max(), val_pairs['pair_idx'].max())
if max_idx >= len(df):
    print(f"   ⚠️ WARNING: Max index {max_idx} >= dataset size {len(df)}")
else:
    print(f"   ✓ All indices within bounds")

# Load features if available
try:
    simple_feats = np.load(data_dir / 'features' / 'places_features_simple.npy')
    pretrained_feats = np.load(data_dir / 'features' / 'places_features_pretrained.npy')
    print(f"\n5. Features:")
    print(f"   Simple: {simple_feats.shape}")
    print(f"   Pretrained: {pretrained_feats.shape}")
    
    # Check for zero vectors
    zero_simple = np.sum(np.all(simple_feats == 0, axis=1))
    zero_pretrained = np.sum(np.all(pretrained_feats == 0, axis=1))
    print(f"   Zero vectors (simple): {zero_simple}")
    print(f"   Zero vectors (pretrained): {zero_pretrained}")
except Exception as e:
    print(f"\n5. Features: Not found or error - {e}")

# Check activity/weather/mood distributions
print(f"\n6. Category distributions:")
print(f"   Activity types: {df['Activity_Type'].nunique()}")
print(df['Activity_Type'].value_counts().head(5))
print(f"\n   Weather types: {df['Weather_Type'].nunique()}")
print(df['Weather_Type'].value_counts())
print(f"\n   Mood categories: {df['Mood_Category'].nunique()}")
print(df['Mood_Category'].value_counts())
