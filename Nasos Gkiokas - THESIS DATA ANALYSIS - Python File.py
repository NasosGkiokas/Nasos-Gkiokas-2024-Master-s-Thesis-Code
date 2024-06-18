#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Import necessary libraries
import pandas as pd
from scipy.stats import ttest_ind, f_oneway, pearsonr
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the survey results, skipping the first two rows
data = pd.read_csv('C:/Users/Lenovo/Desktop/Thesis Survey Results/Masters Thesis Survey June 6 results.csv', skiprows=2)

# Manually set the column names based on the survey structure
column_names = [
    'Age', 'Gender', 'Location', 'Frequency of Social Media Use', 
    'Q5', 'Q6', 'Q7', 'Q8', 'Q9a', 'Q9b', 'Q9c', 
    'Q10', 'Q11', 'Q12', 'Q13', 'Q14'
]
data.columns = column_names

# Consolidate different spellings of countries
data['Location'] = data['Location'].str.strip().str.lower()
country_mapping = {
    'greece': 'Greece',
    'greece athens': 'Greece',
    'greece.': 'Greece',
    'greece': 'Greece',
    'united kingdom': 'United Kingdom',
    'uk': 'United Kingdom',
    'england': 'United Kingdom',
    'united kingdom.': 'United Kingdom',
    'ireland': 'United Kingdom',
    'the netherlands': 'Netherlands',
    'netherlands': 'Netherlands',
    'united states': 'United States',
    'usa': 'United States',
    'miami usa': 'United States',
    'united states.': 'United States',
    'athens': 'Greece',
    'republic of ecuador': 'Ecuador',
    'ecuador': 'Ecuador'
}
data['Location'] = data['Location'].replace(country_mapping)

# Convert Likert scale responses to numeric values
likert_columns = ['Q5', 'Q6', 'Q7', 'Q8', 'Q9a', 'Q9b', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14']
likert_scale = {
    'Strongly disagree': 1,
    'Disagree': 2,
    'Neither agree nor disagree': 3,
    'Agree': 4,
    'Strongly agree': 5,
    'Do not understand at all': 1,
    'Somewhat do not understand': 2,
    'Neither understand nor misunderstand': 3,
    'Somewhat understand': 4,
    'Fully understand': 5,
    'Not at all aligned': 1,
    'Slightly aligned': 2,
    'Moderately aligned': 3,
    'Very aligned': 4,
    'Completely aligned': 5,
    'Significantly decreases trust': 1,
    'Somewhat decreases trust': 2,
    'No effect on trust': 3,
    'Somewhat increases trust': 4,
    'Significantly increases trust': 5,
    'Greatly decreases my willingness': 1,
    'Somewhat decreases my willingness': 2,
    'No effect on my willingness': 3,
    'Somewhat increases my willingness': 4,
    'Greatly increases my willingness': 5,
    'I read the full policy when notified of changes': 5,
    'I skim the policy for key changes': 4,
    'I generally ignore policy updates': 3,
    'I am seldom aware of policy changes': 2
}

# Apply the mapping to the Likert scale columns
for column in likert_columns:
    data[column] = data[column].map(likert_scale)

# Descriptive Statistics
def descriptive_statistics(data):
    return data.describe(include='all')

# Hypothesis 1: T-test
def hypothesis_1(data):
    with_incentives = data[data['Q5'] >= 4]  # Likert scale 4 or 5
    without_incentives = data[data['Q5'] < 4]
    t_stat, p_value = ttest_ind(with_incentives['Q5'], without_incentives['Q5'])
    return t_stat, p_value

# Hypothesis 2: ANOVA
def hypothesis_2(data):
    anova_result = f_oneway(data['Q6'], data['Q7'])
    return anova_result.statistic, anova_result.pvalue

# Hypothesis 3: Correlation
def hypothesis_3(data):
    correlation, p_value = pearsonr(data['Q9a'].dropna(), data['Q8'].dropna())
    return correlation, p_value

# Hypothesis 4: Correlation
def hypothesis_4(data):
    # Drop rows where either Q9b or Q10 is NaN
    df = data[['Q9b', 'Q10']].dropna()
    correlation, p_value = pearsonr(df['Q9b'], df['Q10'])
    return correlation, p_value

# Hypothesis 5: Regression
def hypothesis_5(data):
    df = data[['Q6', 'Q7', 'Q11']].dropna()
    X = df[['Q6', 'Q7']]
    y = df['Q11']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.summary()

# Hypothesis 6: Regression
def hypothesis_6(data):
    df = data[['Q12', 'Q13']].dropna()
    X = df[['Q12']]
    y = df['Q13']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.summary()

# Run analyses
print("Descriptive Statistics:\n", descriptive_statistics(data))
print("\nHypothesis 1 (T-test): T-statistic={}, P-value={}".format(*hypothesis_1(data)))
print("\nHypothesis 2 (ANOVA): F-statistic={}, P-value={}".format(*hypothesis_2(data)))
print("\nHypothesis 3 (Correlation): Correlation={}, P-value={}".format(*hypothesis_3(data)))
print("\nHypothesis 4 (Correlation): Correlation={}, P-value={}".format(*hypothesis_4(data)))
print("\nHypothesis 5 (Regression):\n", hypothesis_5(data))
print("\nHypothesis 6 (Regression):\n", hypothesis_6(data))

# Visualization
def plot_distribution(column, title):
    sns.histplot(data[column], bins=5, kde=False)
    plt.xlabel('Consent Level (Likert Scale)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

plot_distribution('Q5', 'Distribution of Consent Levels for Privacy Policies with Economic Incentives')


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway, pearsonr
import statsmodels.api as sm

# Load the survey results, skipping the first two rows
data = pd.read_csv('C:/Users/Lenovo/Desktop/Thesis Survey Results/Masters Thesis Survey June 6 results.csv', skiprows=2)

# Manually set the column names based on the survey structure
column_names = [
    'Age', 'Gender', 'Location', 'Frequency of Social Media Use', 
    'Q5', 'Q6', 'Q7', 'Q8', 'Q9a', 'Q9b', 'Q9c', 
    'Q10', 'Q11', 'Q12', 'Q13', 'Q14'
]
data.columns = column_names

# Consolidate different spellings of countries
data['Location'] = data['Location'].str.strip().str.lower()
country_mapping = {
    'greece': 'Greece',
    'greece athens': 'Greece',
    'greece.': 'Greece',
    'greece': 'Greece',
    'united kingdom': 'United Kingdom',
    'uk': 'United Kingdom',
    'england': 'United Kingdom',
    'united kingdom.': 'United Kingdom',
    'ireland': 'United Kingdom',
    'the netherlands': 'Netherlands',
    'netherlands': 'Netherlands',
    'united states': 'United States',
    'usa': 'United States',
    'miami usa': 'United States',
    'united states.': 'United States',
    'athens': 'Greece',
    'republic of ecuador': 'Ecuador',
    'ecuador': 'Ecuador'
}
data['Location'] = data['Location'].replace(country_mapping)

# Convert Likert scale responses to numeric values
likert_columns = ['Q5', 'Q6', 'Q7', 'Q8', 'Q9a', 'Q9b', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14']
likert_scale = {
    'Strongly disagree': 1,
    'Disagree': 2,
    'Neither agree nor disagree': 3,
    'Agree': 4,
    'Strongly agree': 5,
    'Do not understand at all': 1,
    'Somewhat do not understand': 2,
    'Neither understand nor misunderstand': 3,
    'Somewhat understand': 4,
    'Fully understand': 5,
    'Not at all aligned': 1,
    'Slightly aligned': 2,
    'Moderately aligned': 3,
    'Very aligned': 4,
    'Completely aligned': 5,
    'Significantly decreases trust': 1,
    'Somewhat decreases trust': 2,
    'No effect on trust': 3,
    'Somewhat increases trust': 4,
    'Significantly increases trust': 5,
    'Greatly decreases my willingness': 1,
    'Somewhat decreases my willingness': 2,
    'No effect on my willingness': 3,
    'Somewhat increases my willingness': 4,
    'Greatly increases my willingness': 5,
    'I read the full policy when notified of changes': 5,
    'I skim the policy for key changes': 4,
    'I generally ignore policy updates': 3,
    'I am seldom aware of policy changes': 2
}

# Apply the mapping to the Likert scale columns
for column in likert_columns:
    data[column] = data[column].map(likert_scale)

# Visualization of Demographics
# Bar chart for age distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Age')
plt.title('Age Distribution of Participants')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Bar chart for gender distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Gender')
plt.title('Gender Distribution of Participants')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.show()

# Bar chart for location distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=data, y='Location', order=data['Location'].value_counts().index)
plt.title('Location Distribution of Participants')
plt.xlabel('Frequency')
plt.ylabel('Location')
plt.show()

# Bar chart for frequency of social media use
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Frequency of Social Media Use')
plt.title('Frequency of Social Media Use')
plt.xlabel('Frequency of Use')
plt.ylabel('Frequency')
plt.show()

# Histogram for Q5
plt.figure(figsize=(10, 6))
sns.histplot(data['Q5'], bins=5, kde=False)
plt.title('Distribution of Consent Levels for Privacy Policies with Economic Incentives')
plt.xlabel('Consent Level (Likert Scale)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot for Hypothesis 3: Understanding data use vs. Perceived ownership
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Q9a', y='Q8')
plt.title('Scatter Plot: Understanding Data Use vs. Perceived Ownership')
plt.xlabel('Understanding Data Use for Economic Gains (Q9a)')
plt.ylabel('Perceived Ownership (Q8)')
plt.show()

# Regression plot for Hypothesis 5: Transparency and trust
plt.figure(figsize=(10, 6))
sns.regplot(data=data, x='Q6', y='Q11')
plt.title('Regression Plot: Perceived Complexity (Q6) and Trust (Q11)')
plt.xlabel('Perceived Complexity (Q6)')
plt.ylabel('Trust (Q11)')
plt.show()

# Regression plot for Hypothesis 6: Clear disclosures and engagement
plt.figure(figsize=(10, 6))
sns.regplot(data=data, x='Q12', y='Q13')
plt.title('Regression Plot: Clear Disclosures (Q12) and Engagement (Q13)')
plt.xlabel('Clear Disclosures (Q12)')
plt.ylabel('Engagement (Q13)')
plt.show()

# Heatmap for correlation matrix (selecting only numeric columns)
plt.figure(figsize=(12, 8))
numeric_columns = data.select_dtypes(include=['number']).columns
correlation_matrix = data[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[ ]:




