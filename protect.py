import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'synthetic_data.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(df.head())

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Tokenize the 'id' and 'name' columns
df['id'] = label_encoder.fit_transform(df['id'])
df['name'] = label_encoder.fit_transform(df['name'])

# Display the first few rows of the DataFrame after tokenization
print(df.head())


def apply_k_anonymity(df, quasi_identifiers, k):
    # Group by the quasi-identifiers and count the occurrences
    grouped = df.groupby(quasi_identifiers).size().reset_index(name='counts')
    
    # Filter groups that have at least k occurrences
    valid_groups = grouped[grouped['counts'] >= k]
    
    # Merge the valid groups back to the original DataFrame
    k_anonymous_df = df.merge(valid_groups[quasi_identifiers], on=quasi_identifiers, how='inner')
    
    return k_anonymous_df

# Define quasi-identifiers and k value
quasi_identifiers = ['Age', 'Gender', 'ZipCode', 'Education', 'Nationality']
k = 3

# Apply k-anonymity
k_anonymous_df = apply_k_anonymity(df, quasi_identifiers, k)
print(k_anonymous_df.head())

def apply_l_diversity(df, quasi_identifiers, sensitive_attr, l):
    # Group by the quasi-identifiers and aggregate the sensitive attribute
    grouped = df.groupby(quasi_identifiers)[sensitive_attr].nunique().reset_index(name='unique_sensitive_values')
    
    # Filter groups that have at least l different sensitive values
    valid_groups = grouped[grouped['unique_sensitive_values'] >= l]
    
    # Merge the valid groups back to the original DataFrame
    l_diverse_df = df.merge(valid_groups[quasi_identifiers], on=quasi_identifiers, how='inner')
    
    return l_diverse_df

# Define sensitive attribute and l value
sensitive_attr = 'Salary'
l = 2

# Apply l-diversity
l_diverse_df = apply_l_diversity(k_anonymous_df, quasi_identifiers, sensitive_attr, l)
print(l_diverse_df.head())

def apply_t_closeness(df, quasi_identifiers, sensitive_attr, t):
    # Calculate the overall distribution of the sensitive attribute
    overall_dist = df[sensitive_attr].value_counts(normalize=True)
    
    # Group by the quasi-identifiers and calculate the distribution of the sensitive attribute within each group
    grouped = df.groupby(quasi_identifiers)[sensitive_attr].value_counts(normalize=True).reset_index(name='proportion')
    
    # Calculate the t-closeness for each group
    def calculate_t_closeness(group):
        group_dist = group.set_index(sensitive_attr)['proportion']
        return (group_dist - overall_dist).abs().max()
    
    t_closeness = grouped.groupby(quasi_identifiers).apply(calculate_t_closeness).reset_index(name='t_closeness')
    
    # Filter groups that satisfy the t-closeness condition
    valid_groups = t_closeness[t_closeness['t_closeness'] <= t]
    
    # Merge the valid groups back to the original DataFrame
    t_close_df = df.merge(valid_groups[quasi_identifiers], on=quasi_identifiers, how='inner')
    
    return t_close_df

# Define t value
t = 0.2

# Apply t-closeness
t_close_df = apply_t_closeness(k_anonymous_df, quasi_identifiers, sensitive_attr, t)
print(t_close_df.head())