
import pandas as pd
import numpy as np

def derive_category(text, keywords_dict, default='Other'):
    """Derive a category based on keywords in the text."""
    if pd.isna(text):
        return default
    text = str(text).lower()
    for category, keywords in keywords_dict.items():
        if any(word in text for word in keywords):
            return category
    return default

def add_derived_columns(df):
    """Add Activity_Type, Weather_Type, and Mood_Category to the dataframe."""
    
    # 1. Activity Type
    activity_keywords = {
        'Nature': ['park', 'garden', 'mountain', 'lake', 'river', 'nature', 'hike', 'hiking', 'forest', 'valley', 'waterfall', 'beach', 'sea', 'ocean', 'island', 'sand', 'cave', 'rock', 'hill', 'view', 'landscape', 'sunrise', 'sunset'],
        'History': ['museum', 'castle', 'palace', 'temple', 'church', 'cathedral', 'history', 'ancient', 'monument', 'ruins', 'art', 'statue', 'shrine', 'mosque', 'tomb', 'archaeology', 'historic'],
        'Urban': ['city', 'street', 'building', 'bridge', 'tower', 'square', 'market', 'shop', 'downtown', 'urban', 'skyline', 'hotel', 'mall', 'road', 'town', 'architecture', 'skyscraper'],
    }
    df['Activity_Type'] = df['Description'].apply(lambda x: derive_category(x, activity_keywords, default='Leisure/Other'))

    # 2. Weather Type
    weather_keywords = {
        'Sunny': ['sunny', 'sun', 'bright', 'clear', 'shine'],
        'Cloudy': ['cloudy', 'cloud', 'gray', 'overcast'],
        'Rainy': ['rain', 'rainy', 'wet', 'storm', 'drizzle'],
        'Snowy': ['snow', 'snowy', 'ice', 'cold', 'winter', 'white'],
        'Windy': ['wind', 'windy', 'breeze', 'blow']
    }
    df['Weather_Type'] = df['Description'].apply(lambda x: derive_category(x, weather_keywords, default='Sunny'))

    # 3. Mood Category
    mood_keywords = {
        'Relaxing': ['relax', 'peace', 'calm', 'quiet', 'serene', 'tranquil', 'rest'],
        'Exciting': ['exciting', 'adventure', 'thrill', 'fun', 'active', 'sport', 'energy'],
        'Romantic': ['romantic', 'love', 'couple', 'date', 'lovely'],
        'Inpsiring': ['inspire', 'awe', 'amazing', 'beautiful', 'wonderful', 'magnificent'],
        'Melancholic': ['sad', 'gloomy', 'lonely', 'dark']
    }
    df['Mood_Category'] = df['Description'].apply(lambda x: derive_category(x, mood_keywords, default='Relaxing'))
    
    return df
