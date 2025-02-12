import pandas as pd

def is_k_anonymous(file_path, quasi_identifiers, k):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Group by the quasi-identifiers and count the occurrences
    grouped = df.groupby(quasi_identifiers).size()
    
    # Check if all groups have at least k occurrences
    k_anonymous = all(grouped >= k)
    print(k_anonymous)
    
    return k_anonymous

# Example usage
# file_path = 'synthetic_data.csv'
# quasi_identifiers = ['Age', 'Gender', 'ZipCode', 'Education', 'Nationality']
# k = 3

# if is_k_anonymous(file_path, quasi_identifiers, k):
#     print(f"The dataset is {k}-anonymous.")
# else:
#     print(f"The dataset is not {k}-anonymous.")

is_k_anonymous('synthetic_data.csv', ["Education"], 3)



# def is_k_anon(dataset, attrs, k):
#     result = False
#     groups ={}
#     for row in dataset:
#         key = []
#         for attr in attrs:
#             key.append(row[attr])

#         if key not in groups:
#             groups[key] = []

#         groups[key].append(row)
#         pass 


#     result = True

#     for group in groups:
#         if len(groups[group]) < k:
#             result = False  
#             break


#     return result

def l_diversity(dataset, sensitive_attr, attrs, l):
    result = False
    groups ={}
    for row in dataset:
        key = []
        for attr in attrs:
            key.append(row[attr])

        if key not in groups:
            groups[key] = []

        groups[key].append(row)
        pass 


    result = True

    for group in groups:
        sensitive_values = []
        for record in groups[group]:
            sensitive_values.append(record[sensitive_attr])

        if len(set(sensitive_values)) < l:
            result = False  
            break


    return result

def l_diversity(file_path, quasi_identifiers, sensitive_attr, l):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Group by the quasi-identifiers and aggregate the sensitive attribute
    grouped = df.groupby(quasi_identifiers)[sensitive_attr].nunique()
    
    # Check if all groups have at least l different sensitive values
    l_diverse = all(grouped >= l)
    
    return l_diverse

# Example usage
file_path = 'synthetic_data.csv'
quasi_identifiers = ['Age', 'Gender', 'ZipCode', 'Education', 'Nationality']
sensitive_attr = 'Salary'
l = 2

if l_diversity(file_path, quasi_identifiers, sensitive_attr, l):
    print(f"The dataset is {l}-diverse for the column(s) {quasi_identifiers} with respect to {sensitive_attr}.")
else:
    print(f"The dataset is not {l}-diverse for the column(s) {quasi_identifiers} with respect to {sensitive_attr}.") 

def t_closeness(file_path, quasi_identifiers, sensitive_attr, t):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Calculate the overall distribution of the sensitive attribute
    overall_dist = df[sensitive_attr].value_counts(normalize=True)
    
    # Group by the quasi-identifiers and calculate the distribution of the sensitive attribute within each group
    grouped = df.groupby(quasi_identifiers)[sensitive_attr].value_counts(normalize=True)
    
    # Calculate the t-closeness for each group
    t_closeness = grouped.groupby(level=0).apply(lambda x: (x - overall_dist).abs().max())
    
    # Check if all groups satisfy the t-closeness condition
    t_close = all(t_closeness <= t)
    
    return t_close

