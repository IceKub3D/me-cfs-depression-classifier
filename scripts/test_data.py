import pandas as pd
test_df = pd.read_csv('data/test_data.csv')
print(test_df[['gender', 'work_status', 'social_activity_level', 'exercise_frequency', 'meditation_or_mindfulness']].head())
print(test_df.dtypes)