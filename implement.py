# import pandas as pd
# import numpy as np
# from collections import defaultdict

# # ------------------------- Data Preprocessing -------------------------
# # Generalization rules for categorical attributes
# def generalize_education(education):
#     mapping = {
#         "Middle School": "Secondary Education",
#         "High School": "Secondary Education",
#         "Associate Degree": "Higher Education",
#         "Bachelor's": "Higher Education",
#         "Master's": "Advanced Education",
#         "PhD": "Advanced Education"
#     }
#     return mapping.get(education, education)

# def generalize_age(age):
#     age_bins = [(0, 18), (19, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 100)]
#     for (low, high) in age_bins:
#         if low <= age <= high:
#             return f"{low}-{high}"
#     return "Unknown"

# def generalize_zipcode(zipcode):
#     if 15500 <= zipcode < 16000:
#         return "15500-16000"
#     elif 16000 <= zipcode < 16500:
#         return "16000-16500"
#     elif 16500 <= zipcode < 17000:
#         return "16500-17000"
#     elif 17000 <= zipcode < 17500:
#         return "17000-17500"
#     elif 17500 <= zipcode < 18000:
#         return "17500-18000"
#     elif 18000 <= zipcode < 18500:
#         return "18000-18500"
#     elif 18500 <= zipcode < 19000:
#         return "18500-19000"
#     elif 19000 <= zipcode < 19500:
#         return "19000-19500"
#     elif 19500 <= zipcode < 19999:
#         return "19500-19999"
#     else:
#         return "Unknown"

# def generalize_country(country):
#     continent_mapping = {
#         'Africa': ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'],
#         'Asia': ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia', 'China', 'Cyprus', 'Georgia', 'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal', 'North Korea', 'Oman', 'Pakistan', 'Palestine', 'Philippines', 'Qatar', 'Saudi Arabia', 'Singapore', 'South Korea', 'Sri Lanka', 'Syria', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen'],
#         'Europe': ['Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Georgia', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kazakhstan', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'Ukraine', 'United Kingdom', 'Vatican City'],
#         'North America': ['Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic', 'El Salvador', 'Grenada', 'Guatemala', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Trinidad and Tobago', 'United States'],
#         'Oceania': ['Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'New Zealand', 'Palau', 'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu'],
#         'South America': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela']
#     }
#     for continent, countries in continent_mapping.items():
#         if country in countries:
#             return continent
#     return 'Unknown'

# # ------------------------- Mondrian Algorithm -------------------------
# def choose_split_attribute(data, quasi_identifiers):
#     """ Choose the attribute with the widest normalized range """
#     ranges = {}
#     for attr in quasi_identifiers:
#         if data[attr].dtype == 'object':
#             ranges[attr] = len(data[attr].unique())  # Categorical: Count unique values
#         else:
#             ranges[attr] = data[attr].max() - data[attr].min()  # Numerical: Range size
#     return max(ranges, key=ranges.get)

# def partition_data(data, quasi_identifiers, k):
#     """ Recursively partition data while maintaining k-anonymity """
#     if len(data) < k:
#         return [data]
    
#     best_attr = choose_split_attribute(data, quasi_identifiers)
#     if data[best_attr].dtype == 'object':
#         sorted_values = sorted(data[best_attr].unique())
#         median_idx = len(sorted_values) // 2
#         split_val = sorted_values[median_idx] 
#         left = data[data[best_attr] <= split_val]
#         right = data[data[best_attr] > split_val]
#     else:
#         median_val = data[best_attr].median()
#         left = data[data[best_attr] <= median_val]
#         right = data[data[best_attr] > median_val]
    
#     if len(left) < k or len(right) < k:
#         return [data]
    
#     return partition_data(left, quasi_identifiers, k) + partition_data(right, quasi_identifiers, k)

# def anonymize_numerical(partition, attr):
#     """ Generalize numerical attribute in partition """
#     min_val = partition[attr].min()
#     max_val = partition[attr].max()
#     partition[attr] = f"{min_val}-{max_val}"
#     return partition

# # def anonymize_categorical(partition, attr):
# #     """ Generalize categorical attribute in partition """
# #     unique_values = partition[attr].unique()
# #     if len(unique_values) == 1:
# #         partition[attr] = unique_values[0]
# #     else:
# #         partition[attr] = partition[attr].apply(lambda x: f"{partition[attr].min()} - {partition[attr].max()}")
# #     return partition

# def anonymize_categorical(partition, attr, mapping):
#     """
#     Generalizes categorical attribute in partition using predefined mapping.
    
#     Parameters:
#     - partition: DataFrame containing the dataset
#     - attr: The categorical attribute (column) to be generalized
#     - mapping: A dictionary where the key is the original category and the value is the generalized category
    
#     Returns:
#     - The partition DataFrame with generalized values
#     """
#     # Check if the attribute exists in the partition
#     if attr not in partition.columns:
#         raise ValueError(f"The attribute '{attr}' does not exist in the dataset.")
    
#     # Apply the mapping to the attribute column
#     partition[attr] = partition[attr].map(mapping).fillna(partition[attr])  # Keep original value if no mapping exists
    
#     return partition

# def anonymize_partition(partition, quasi_identifiers):
#     """ Generalize partitioned data """
#     for attr in quasi_identifiers:
#         if attr == "Education":
#             mapping = {
#                 "Middle School": "Secondary Education",
#                 "High School": "Secondary Education",
#                 "Associate Degree": "Higher Education",
#                 "Bachelor": "Higher Education",
#                 "Master": "Advanced Education",
#                 "PhD": "Advanced Education"
#             }
#             partition = anonymize_categorical(partition, attr, mapping)
#         elif attr == "Nationality":
#             continent_mapping = {
#                 'Africa': ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'],
#                 'Asia': ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia', 'China', 'Cyprus', 'Georgia', 'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal', 'North Korea', 'Oman', 'Pakistan', 'Palestine', 'Philippines', 'Qatar', 'Saudi Arabia', 'Singapore', 'South Korea', 'Sri Lanka', 'Syria', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen'],
#                 'Europe': ['Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Georgia', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kazakhstan', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'Ukraine', 'United Kingdom', 'Vatican City'],
#                 'North America': ['Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic', 'El Salvador', 'Grenada', 'Guatemala', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Trinidad and Tobago', 'United States'],
#                 'Oceania': ['Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'New Zealand', 'Palau', 'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu'],
#                 'South America': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela']
#             }
#             mapping = {country: continent for continent, countries in continent_mapping.items() for country in countries}
#             partition = anonymize_categorical(partition, attr, mapping)
#         elif partition[attr].dtype == 'object':
#             partition = anonymize_categorical(partition, attr, {})
#         else:
#             partition = anonymize_numerical(partition, attr)
#     return partition

# def k_anonymize(data, quasi_identifiers, k):
#     """ Perform Mondrian K-Anonymization """
#     partitions = partition_data(data, quasi_identifiers, k)
#     anonymized_data = pd.concat([anonymize_partition(part, quasi_identifiers) for part in partitions])
#     return anonymized_data

# # ------------------------- Example Execution -------------------------
# # Sample dataset
# # data = pd.DataFrame({
# #     "Age": [23, 45, 34, 19, 50, 37, 28, 41, 32, 60],
# #     "Education": ["High School", "Bachelor's", "Master's", "High School", "PhD", "Bachelor's", "Associate Degree", "Master's", "PhD", "Middle School"],
# #     "Nationality": ["USA", "Germany", "India", "Canada", "Brazil", "UK", "China", "Mexico", "Argentina", "South Africa"],
# #     "ZipCode": [15700, 15800, 15900, 16000, 16100, 16200, 16300, 16400, 16500, 16600]
# # })


# data = pd.read_csv('relaxed_anonymized_data.csv')
# # Define quasi-identifiers
# quasi_identifiers = ["Age", "Education", "Nationality", "ZipCode"]

# # Perform k-anonymity (k=2)
# anonymized_data = k_anonymize(data, quasi_identifiers, k=2)
# print(anonymized_data)

import matplotlib.pyplot as plt

# Provided data
dataset_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
nac_values = [1.6666666666666667, 1.6666666666666667, 1.6666666666666667, 1.3333333333333333, 1.3175230566534915, 1.5475085112968123, 1.4690759512266782]
info_loss_values = [11.249999999999998, 2.7445460942997895, 1.8513513513513509, 1.3279720649701068, 1.2753781169096565, 1.2361677580799404, 1.2319623943251474]
execution_time_values = [0.0, 0.024586915969848633, 0.026307106018066406, 0.3982830047607422, 0.4151453971862793, 1.5818188190460205, 4.506495237350464]

# Plotting NAC vs Dataset Size
plt.figure(figsize=(12, 8))
plt.plot(dataset_sizes, nac_values, marker='o', color='tab:blue', label='NAC')
plt.xlabel('Dataset Size')
plt.ylabel('NAC')
plt.title('NAC vs Dataset Size')
plt.grid()
plt.legend()
plt.savefig('data/nac_vs_dataset_size_plot.jpg')
plt.show()

# Plotting Information Loss vs Dataset Size
plt.figure(figsize=(12, 8))
plt.plot(dataset_sizes, info_loss_values, marker='o', color='tab:red', label='Information Loss (%)')
plt.xlabel('Dataset Size')
plt.ylabel('Information Loss (%)')
plt.title('Information Loss vs Dataset Size')
plt.grid()
plt.legend()
plt.savefig('data/info_loss_vs_dataset_size_plot.jpg')
plt.show()

# Plotting Execution Time vs Dataset Size
plt.figure(figsize=(12, 8))
plt.plot(dataset_sizes, execution_time_values, marker='o', color='tab:green', label='Execution Time (seconds)')
plt.xlabel('Dataset Size')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Dataset Size')
plt.grid()
plt.legend()
plt.savefig('data/execution_time_vs_dataset_size_plot.jpg')
plt.show()


