import pandas as pd

# Load the datasets
print("Loading datasets...", flush=True)
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Add label columns
# 0 = Fake, 1 = True
fake_df['fake/true(0/1)'] = 0
true_df['fake/true(0/1)'] = 1
print(f"\nAdded labels \nFake: 0, True: 1", flush=True)

# Limit to 1500 each for even distribution and max 3000 rows
fake_df = fake_df.sample(n=1500, random_state=42).reset_index(drop=True)
true_df = true_df.sample(n=1500, random_state=42).reset_index(drop=True)
print("\nSampled 1500 rows from each dataset", flush=True)

# Check missing values
print("\nMissing values in Fake.csv:", flush=True)
print(fake_df.isna().sum()[fake_df.isna().sum() > 0], flush=True)

print("\nMissing values in True.csv:", flush=True)
print(true_df.isna().sum()[true_df.isna().sum() > 0], flush=True)

# Merge the datasets
merged_df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)
print(f"\nMerged dataset shape: {merged_df.shape}", flush=True)

# Shuffle the data to ensure random distribution
# This is important because we concatenated all fake news first, then all true news
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
print("\nData shuffled randomly")

#group the subjects into 3 categories: Politics & Gov, US & General News, World News
merged_df['subject'] = merged_df['subject'].replace(['politics', 'politicsNews', 'Government News', 'left-news'], 'Politics & Gov')
merged_df['subject'] = merged_df['subject'].replace(['US_News', 'News'], 'US & General News')
merged_df['subject'] = merged_df['subject'].replace(['Middle-east', 'worldnews'], 'World News')

merged_df.to_csv('data.csv', index=False)