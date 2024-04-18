import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv("train2023.csv")
disease_columns = data.columns[7:19]
for column in disease_columns:
    data[column] = data[column].apply(lambda x: 1 if x == 1.0 else 0)

disease_counts = data[disease_columns].sum()
print(disease_counts)

# plt.figure(figsize=(12, 6))
# disease_counts.plot(kind='bar', color='skyblue')
# plt.title('Frequency of Each Lung Pathology Where 1 is Recorded')
# plt.xlabel('Lung Pathology')
# plt.ylabel('Count of Positive Occurrences')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

data2 = pd.read_csv("train2023.csv")
disease_columns = data2.columns[7:19]
for column in disease_columns:
    data2[column] = data2[column].apply(lambda x: 1 if x == -1.0 else 0)

disease_counts = data2[disease_columns].sum()

# plt.figure(figsize=(12, 6))
# disease_counts.plot(kind='bar', color='purple')
# plt.title('Frequency of Each Lung Pathology Where -1 is Recorded')
# plt.xlabel('Lung Pathology')
# plt.ylabel('Count of Negative Occurrences')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

data = pd.read_csv("train2023.csv")
disease_columns = data.columns[8:19]

# count num of diseases with a 1 present
data['Any Disease'] = data[disease_columns].fillna(0).sum(axis=1)
data['Any Disease'] = data['Any Disease'].apply(lambda x: 1 if x > 0 else 0)

data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
data.dropna(subset=['Age', 'Any Disease'], inplace=True)

bins = np.arange(0, 111, 10)
labels = [f"{i} - {i+9}" for i in bins[:-1]]
data['Age Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
age_group_frequency = data.groupby('Age Group')['Any Disease'].sum()

# plt.figure(figsize=(12, 6))
# age_group_frequency.plot(kind='bar', color='teal')
# plt.title('Frequency of Any Pathology Presence of 1 by Age Group')
# plt.xlabel('Age Group')
# plt.ylabel('Frequency of Pathology Presence')
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--')
# plt.show()

age_group_counts = data['Age Group'].value_counts().sort_index()

# plt.figure(figsize=(12, 8))
# age_group_counts.plot(kind='bar', color='skyblue')
# plt.title('Number of People per Age Group')
# plt.xlabel('Age Group')
# plt.ylabel('Number of People')
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--')
# plt.show()

data = pd.read_csv("train2023.csv")
disease_columns = data.columns[8:19]

counts_dict = {col: data[col].value_counts(dropna=False) for col in disease_columns}
counts_df = pd.DataFrame(counts_dict).fillna(0).astype(int)
counts_df = counts_df.reindex([1, -1, 0, np.nan]).rename(index={np.nan: 'NaN'})

color_map = {
    '1.0': 'green',    
    '-1.0': 'red',     
    '0.0': 'blue',     
    'NaN': 'gray'    
}

fig, ax = plt.subplots(figsize=(12, 6))
positions = list(range(len(disease_columns)))  
width = 0.2

for i, category in enumerate(counts_df.index):
    category_key = f'{category}' if pd.notna(category) else 'NaN'
    category_color = color_map[category_key]
    ax.bar([p + width*i for p in positions], counts_df.loc[category], width=width, label=str(category), color=category_color)

# ax.set_title('Count of 1, -1, 0, and NaN for Each Pathology')
# ax.set_xlabel('Pathology')
# ax.set_ylabel('Count')
# ax.set_xticks([p + width*1.5 for p in positions])
# ax.set_xticklabels(disease_columns, rotation=45)
# ax.legend(title='Value')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

data = pd.read_csv("train2023.csv")
if 'LL' in data.columns:
    data = data.drop('LL', axis=1)
view_counts = data['AP/PA'].value_counts()
plt.figure(figsize=(8, 6))
view_counts.plot(kind='bar', color=['blue', 'green'])
plt.title('Frequency of AP vs. PA Views')
plt.xlabel('View Type')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()