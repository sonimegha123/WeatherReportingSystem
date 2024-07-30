# Databricks notebook source
!pip install jsonlines


# COMMAND ----------

!pip install simpledorff

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import pandas as pd

def extract_labels(data):
    rows = []
    for row in data:
        data_row = row.get("data_row", {})
        data_row_id = data_row.get("id", "")
        projects = row.get("projects", {})
        for project_id, project_data in projects.items():
            labels = project_data.get("labels", [])
            for label in labels:
                try:
                    label_info = {
                        "data_row_id": data_row_id,
                        "project_id": project_id,
                        "project_name": project_data["name"],
                        "label_id": label["id"],
                        "label_created_at": label["label_details"]["created_at"],
                        "label_created_by": label["label_details"]["created_by"],
                        "consensus_score": label["performance_details"]["consensus_score"]
                    }
                except:
                    label_info = {
                        "data_row_id": data_row_id,
                        "project_id": project_id,
                        "project_name": project_data["name"],
                        "label_id": label["id"],
                        "label_created_at": label["label_details"]["created_at"],
                        "label_created_by": label["label_details"]["created_by"],
                        "consensus_score": None
                    }
                
                annotations = label.get("annotations", {})
                
                # Capture direct annotations
                for annotation in annotations.get("classifications", []):
                    process_classification(annotation, label_info)
                
                rows.append(label_info)
    return rows

def process_classification(annotation, label_info):
    annotation_name = annotation.get("name")
    radio_answer = annotation.get("radio_answer", {})
    checklist_answers = annotation.get("checklist_answers", [])
    
    if annotation_name:
        # Capture radio answer if it exists
        if radio_answer:
            label_info[annotation_name] = radio_answer.get("name", "")
        
        # Capture checklist answers if they exist
        if checklist_answers:
            checklist_values = [answer.get("name", "") for answer in checklist_answers]
            label_info[annotation_name] = ", ".join(checklist_values)
        
        # Handle nested classifications within radio answers
        nested_classifications = radio_answer.get("classifications", [])
        for nested_annotation in nested_classifications:
            process_classification(nested_annotation, label_info)
        
        # Handle nested classifications directly under the current classification
        nested_classifications = annotation.get("classifications", [])
        for nested_annotation in nested_classifications:
            process_classification(nested_annotation, label_info)


file_path = '/dbfs/mnt/ds-data-apps/megha/AnnotationLab_data/Buienradar_labels.ndjson'

def parse(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, start=1):
            try:
                data_row = json.loads(line.strip())
                data.append(data_row)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
    return data

def create_dataframe(data):
    df = pd.DataFrame(data)
    return df

data = parse(file_path)
labeling_results = extract_labels(data)
df = create_dataframe(labeling_results)


display(df)


# COMMAND ----------

# MAGIC %md
# MAGIC #Cleaning the dataframe

# COMMAND ----------

df = df.rename(columns={
    'data_row_id': 'DataRowID',
    'project_id': 'ProjectID',
    'project_name': 'ProjectName',
    'label_id': 'LabelID',
    'label_created_at': 'LabelCreatedAt',
    'label_created_by': 'LabelCreatedBy',
    'consensus_score': 'ConsensusScore',
    'Wordt er informatie gegeven over deze ochtend?': 'MorningInfoGiven',
    'Welke weerinformatie wordt gegeven over deze ochtend?': 'MorningWeatherInfo',
    'Hoe correct is de informatie over deze ochtend op een schaal van 1 tot 5?': 'MorningCorrectness',
    'Wordt er informatie gegeven over deze middag?': 'AfternoonInfoGiven',
    'Welke weerinformatie wordt gegeven over deze middag?': 'AfternoonWeatherInfo',
    'Hoe correct is de informatie over deze middag op een schaal van 1 tot 5?': 'AfternoonCorrectness',
    'Wordt er informatie gegeven over deze avond?': 'EveningInfoGiven',
    'Welke weerinformatie wordt gegeven over deze avond?': 'EveningWeatherInfo',
    'Hoe correct is de informatie over deze avond op een schaal van 1 tot 5?': 'EveningCorrectness',
    'Wordt er informatie gegeven over morgen?': 'TomorrowInfoGiven',
    'Welke weerinformatie wordt gegeven over morgen?': 'TomorrowWeatherInfo',
    'Hoe correct is de informatie over morgen op een schaal van 1 tot 5?': 'TomorrowCorrectness',
    'Zou het je opvallen dat dit weerbericht automatisch is gegenereerd als je het zou lezen op de buienradar website? Op een schaal van 1-5': 'AutoGeneratedNoticeable',
    'Is de opzet van het weerbericht goed? Oftewel, begint het met deze ochtend, vervolgens de middag, de avond en de nacht en eindigt het met de volgende dag? Op een schaal van 1-5': 'ReportStructureGood',
    'Vind je dit een goed weerbericht, qua taal en begrijpelijkheid, op een schaal van 1 tot 5?': 'LanguageClarity',
    'Hoe tevreden zou je zijn met dit weerbericht, op een schaal van 1 tot 5?': 'Satisfaction',
    'Wordt er informatie gegeven over aankomende nacht?': 'NightInfoGiven',
    'Welke weerinformatie wordt gegeven over aankomende nacht?': 'NightWeatherInfo',
    'Hoe correct is deze informatie op een schaal van 1 tot 5?': 'NightCorrectness'
})


# COMMAND ----------

import re

def clean_correctness_column(value):
    if isinstance(value, str):
        match = re.search(r'\b[1-5]\b', value)
        return int(match.group()) if match else None
    elif isinstance(value, (int, float)) and not pd.isna(value):
        return int(value)
    else:
        return None


df['MorningCorrectness'] = df['MorningCorrectness'].apply(clean_correctness_column)



# COMMAND ----------

df['AfternoonCorrectness'] = df['AfternoonCorrectness'].apply(clean_correctness_column)
df['EveningCorrectness'] = df['EveningCorrectness'].apply(clean_correctness_column)
df['TomorrowCorrectness'] = df['TomorrowCorrectness'].apply(clean_correctness_column)
df['AutoGeneratedNoticeable'] = df['AutoGeneratedNoticeable'].apply(clean_correctness_column)
df['ReportStructureGood'] = df['ReportStructureGood'].apply(clean_correctness_column)
df['LanguageClarity'] = df['LanguageClarity'].apply(clean_correctness_column)
df['Satisfaction'] = df['Satisfaction'].apply(clean_correctness_column)
df['NightCorrectness'] = df['NightCorrectness'].apply(clean_correctness_column)

# COMMAND ----------

def convert_column_to_integer(df, column_name):
    try:
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce').astype(pd.Int64Dtype())
    except ValueError:
        print(f"Error converting column '{column_name}' to integer.")

convert_column_to_integer(df, 'MorningCorrectness')
convert_column_to_integer(df, 'AfternoonCorrectness')
convert_column_to_integer(df, 'EveningCorrectness')
convert_column_to_integer(df, 'TomorrowCorrectness')
convert_column_to_integer(df, 'AutoGeneratedNoticeable')
convert_column_to_integer(df, 'ReportStructureGood')
convert_column_to_integer(df, 'LanguageClarity')
convert_column_to_integer(df, 'Satisfaction')
convert_column_to_integer(df, 'NightCorrectness')

# COMMAND ----------

excluded_emails = [
'annotatiepilot+demo@gmail.com',
'annotatiepilot+craig@gmail.com',
# 'annotatiepilot+minnes@gmail.com',
# 'annotatiepilot+ali@gmail.com'
]


df = df[~df['LabelCreatedBy'].isin(excluded_emails)]
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #Analysis

# COMMAND ----------

import math

# Average correctness ratings
avg_correctness_morning = df['MorningCorrectness'].mean()
avg_correctness_afternoon = df['AfternoonCorrectness'].mean()
avg_correctness_evening = df['EveningCorrectness'].mean()
avg_correctness_tomorrow = df['TomorrowCorrectness'].mean()

avg_AutoGeneratedNoticeable = df['AutoGeneratedNoticeable'].mean()
avg_ReportStructureGood = df['ReportStructureGood'].mean()
avg_LanguageClarity = df['LanguageClarity'].mean()
avg_Satisfaction = df['Satisfaction'].mean()

avg_ConsensusScore = df['ConsensusScore'].mean()


print("avg_correctness_morning", math.ceil(avg_correctness_morning))
print("avg_correctness_afternoon", math.ceil(avg_correctness_afternoon))
print("avg_correctness_evening", math.ceil(avg_correctness_evening))
print("avg_correctness_tomorrow", math.ceil(avg_correctness_tomorrow))

print("avg_AutoGeneratedNoticeable", math.ceil(avg_AutoGeneratedNoticeable))
print("avg_ReportStructureGood", math.ceil(avg_ReportStructureGood))
print("avg_LanguageClarity", math.ceil(avg_LanguageClarity))
print("avg_Satisfaction", math.ceil(avg_Satisfaction))

print("avg_ConsensusScore", avg_ConsensusScore)



# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

columns = ['MorningInfoGiven', 'AfternoonInfoGiven', 'EveningInfoGiven', 'TomorrowInfoGiven']
percentage_data = {}

for column in columns:
    percentage_data[column] = df[column].value_counts(normalize=True) * 100

# Create a DataFrame for the calculated percentages
percentage_df = pd.DataFrame(percentage_data).transpose()

# Ensure both 'yes' and 'no' categories are present for each column
percentage_df = percentage_df.fillna(0)

sns.set(style="whitegrid")
percentage_df.plot(kind='bar', figsize=(12, 7), color=['#4CAF50', '#FF5722'], edgecolor='black')

plt.xlabel('Columns', fontsize=14, fontweight='bold')
plt.ylabel('Percentage (%)', fontsize=14, fontweight='bold')
plt.title('Percentage of People by Response in Each Column', fontsize=16, fontweight='bold')

plt.xticks(range(len(percentage_df.index)), percentage_df.index, fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True, which='both', linestyle='--', linewidth=0.5)

for i, column in enumerate(columns):
    for j, val in enumerate(percentage_df.columns):
        height = percentage_df.loc[column, val]
        plt.text(j + i - 0.2, height + 1, f'{height:.2f}%', ha='center', fontsize=12, fontweight='bold')

plt.gca().set_facecolor('#f7f7f7')
plt.legend(title='Responses', title_fontsize='13', fontsize='12')
plt.show()



# COMMAND ----------

# Calculate the percentage distribution for each column
def calculate_percentage_distribution(column):
    return df[column].value_counts(normalize=True).sort_index() * 100

# Find the maximum percentage to set the y-axis limit
max_percentage = 0
for column in ['MorningCorrectness', 'AfternoonCorrectness', 'EveningCorrectness', 'TomorrowCorrectness']:
    max_percentage = max(max_percentage, calculate_percentage_distribution(column).max())

# Create a bar plot for each column using Seaborn
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

columns = ['MorningCorrectness', 'AfternoonCorrectness', 'EveningCorrectness', 'TomorrowCorrectness']
titles = ['Morning Correctness', 'Afternoon Correctness', 'Evening Correctness', "Tomorrow's Correctness"]
axes = axes.flatten()

for ax, column, title in zip(axes, columns, titles):
    percentages = calculate_percentage_distribution(column)
    sns.barplot(x=percentages.index.astype(int), y=percentages.values, ax=ax, palette="viridis")
    ax.set_title(title, fontsize=15)
    ax.set_xlabel('Ratings', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_ylim(0, max_percentage + 10)  
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), 
                    textcoords='offset points')

plt.tight_layout()
plt.show()

# COMMAND ----------

df['Satisfaction'] = df['Satisfaction'].dropna().astype(int)

rating_counts = df['Satisfaction'].value_counts(normalize=True) * 100
rating_counts = rating_counts.sort_index()

sns.set(style="whitegrid")

plt.figure(figsize=(12, 7))
bars = plt.bar(rating_counts.index, rating_counts, color=sns.color_palette("Blues_r", len(rating_counts)), edgecolor='black', width=0.6)

plt.xlabel('Satisfaction Ratings', fontsize=14, fontweight='bold')
plt.ylabel('Percentage of Annotators', fontsize=14, fontweight='bold')
plt.title('Percentage of Annotators by Satisfaction Ratings', fontsize=16, fontweight='bold')

plt.xticks(rating_counts.index, fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True, which='both', linestyle='--', linewidth=0.5)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}%', ha='center', fontsize=12, fontweight='bold')

# Set y-axis limit higher than the maximum bar height
plt.ylim(0, rating_counts.max() + 5)

plt.gca().set_facecolor('#f7f7f7')
plt.show()




# COMMAND ----------

df['AutoGeneratedNoticeable'] = df['AutoGeneratedNoticeable'].dropna().astype(int)

# Calculate the percentage of each rating
noticeable_counts = df['AutoGeneratedNoticeable'].value_counts(normalize=True) * 100
noticeable_counts = noticeable_counts.sort_index()

sns.set(style="whitegrid")

plt.figure(figsize=(12, 7))
bars = plt.bar(noticeable_counts.index, noticeable_counts, color=sns.color_palette("Greens_r", len(noticeable_counts)), edgecolor='black', width=0.6)


plt.xlabel('Auto-Generated Noticeable Ratings', fontsize=14, fontweight='bold')
plt.ylabel('Percentage of Annotators', fontsize=14, fontweight='bold')
plt.title('Percentage of Annotators by Auto-Generated Noticeable Ratings', fontsize=16, fontweight='bold')


plt.xticks(noticeable_counts.index, fontsize=12)
plt.yticks(fontsize=12)


plt.grid(True, which='both', linestyle='--', linewidth=0.5)


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}%', ha='center', fontsize=12, fontweight='bold')

plt.ylim(0, rating_counts.max() + 5)

plt.gca().set_facecolor('#f7f7f7')
plt.show()


# COMMAND ----------

df['ReportStructureGood'] = df['ReportStructureGood'].dropna().astype(int)

# Calculate the percentage of each rating
report_structure_counts = df['ReportStructureGood'].value_counts(normalize=True) * 100
report_structure_counts = report_structure_counts.sort_index()

sns.set(style="whitegrid")


plt.figure(figsize=(12, 7))
bars = plt.bar(report_structure_counts.index, report_structure_counts, color=sns.color_palette("Purples_r", len(report_structure_counts)), edgecolor='black', width=0.6)


plt.xlabel('Report Structure Ratings', fontsize=14, fontweight='bold')
plt.ylabel('Percentage of Annotators', fontsize=14, fontweight='bold')
plt.title('Percentage of Annotators by Report Structure Ratings', fontsize=16, fontweight='bold')

plt.xticks(report_structure_counts.index, fontsize=12)
plt.yticks(fontsize=12)


plt.grid(True, which='both', linestyle='--', linewidth=0.5)


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}%', ha='center', fontsize=12, fontweight='bold')



plt.gca().set_facecolor('#f7f7f7')
plt.show()


# COMMAND ----------

df['LanguageClarity'] = df['LanguageClarity'].dropna().astype(int)

# Calculate the percentage of each rating
report_structure_counts = df['LanguageClarity'].value_counts(normalize=True) * 100
report_structure_counts = report_structure_counts.sort_index()

sns.set(style="whitegrid")


plt.figure(figsize=(12, 7))
bars = plt.bar(report_structure_counts.index, report_structure_counts, color=sns.color_palette("Oranges_r", len(report_structure_counts)), edgecolor='black', width=0.6)


plt.xlabel('Language Clarity Ratings', fontsize=14, fontweight='bold')
plt.ylabel('Percentage of Annotators', fontsize=14, fontweight='bold')
plt.title('Percentage of Annotators by Language Clarity Ratings', fontsize=16, fontweight='bold')

plt.xticks(report_structure_counts.index, fontsize=12)
plt.yticks(fontsize=12)


plt.grid(True, which='both', linestyle='--', linewidth=0.5)


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}%', ha='center', fontsize=12, fontweight='bold')


plt.gca().set_facecolor('#f7f7f7')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # Consensus Score: SimpleDorff - Calculate Krippendorff's Alpha on a DataFrame

# COMMAND ----------

import simpledorff

class_cols = [
    'AutoGeneratedNoticeable',
    'ReportStructureGood',
    'LanguageClarity',
    'Satisfaction',
    'MorningCorrectness',
    'AfternoonCorrectness',
    'EveningCorrectness',
    'NightCorrectness',
    'TomorrowCorrectness'
]

for col in class_cols:
    # Check if the column has at least two unique values
    if df_copy[col].nunique() > 1:
        try:
            alpha = simpledorff.calculate_krippendorffs_alpha_for_df(
                df_copy,
                experiment_col='DataRowID',
                annotator_col='LabelCreatedBy',
                class_col=col
            )
            print(f"Krippendorff's Alpha for {col}: {alpha}")
        except ZeroDivisionError:
            print(f"Krippendorff's Alpha for {col}: Calculation error due to insufficient data")
    else:
        print(f"Krippendorff's Alpha for {col}: Not enough unique values to calculate")


# COMMAND ----------

# MAGIC %md
# MAGIC #Consensus after excluding people with most disagreement 

# COMMAND ----------

excluded_emails = [
'annotatiepilot+pols@gmail.com',
'annotatiepilot+ali@gmail.com'
]


df['annotators'] = df['LabelCreatedBy'].apply(lambda x: None if x in excluded_emails else x)

# COMMAND ----------

for col in class_cols:
    # Check if the column has at least two unique values
    if df_copy[col].nunique() > 1:
        try:
            alpha = simpledorff.calculate_krippendorffs_alpha_for_df(
                df_copy,
                experiment_col='DataRowID',
                annotator_col='annotators',
                class_col=col
            )
            print(f"Krippendorff's Alpha for {col}: {alpha}")
        except ZeroDivisionError:
            print(f"Krippendorff's Alpha for {col}: Calculation error due to insufficient data")
    else:
        print(f"Krippendorff's Alpha for {col}: Not enough unique values to calculate")

