import pandas as pd
import hashlib
import time
import matplotlib.pyplot as plt

class Mondrian:
    def __init__(self, k: int, df: pd.DataFrame, qi: list, decoding_dict=None):
        self.k = k
        self.df = df.copy()
        self.qi = qi
        self.partitions = []  # Stores anonymized partitions
        self.decoding_dict = decoding_dict if decoding_dict else {}

    def choose_dimension(self, examined_qi):
        """
        Chooses the dimension with the widest range of values.
        """
        dim_ranges = {}
        for q in self.qi:
            if pd.api.types.is_numeric_dtype(self.df[q]):
                dim_ranges[q] = self.df[q].max() - self.df[q].min()
            else:
                dim_ranges[q] = len(self.df[q].unique())
        sorted_dims = sorted(dim_ranges.items(), key=lambda x: x[1], reverse=True)
        return sorted_dims[examined_qi][0]

    def anonymize_aux(self):
        self.anonymize(self.df)
        self.generalize_region()

    def anonymize(self, partition, examined_qi=0):
        """
        Greedy partitioning algorithm
        """
        if len(partition) <= 2 * self.k or examined_qi >= len(self.qi):
            self.partitions.append(partition)
            return

        dim = self.choose_dimension(examined_qi)
        if pd.api.types.is_numeric_dtype(partition[dim]):
            fs = partition[dim].values
            split_val = self.find_median(fs)
            lhs_rhs = self.create_partition(partition, dim, split_val)
        else:
            lhs_rhs = None

        if lhs_rhs:
            self.anonymize(lhs_rhs[0])
            self.anonymize(lhs_rhs[1])
        else:
            self.anonymize(partition, examined_qi + 1)

    def create_partition(self, partition, dim, split_val):
        """
        Splits partition into two sub-partitions based on split_val.
        """
        left_partition = partition[partition[dim] <= split_val]
        right_partition = partition[partition[dim] > split_val]

        if len(left_partition) < self.k or len(right_partition) < self.k:
            return None

        return left_partition, right_partition

    def generalize_region(self):
        """
        Generalizes each partition by replacing values with ranges.
        """
        generalized_partitions = []
        for partition in self.partitions:
            for q in self.qi:
                if q == "Nationality":
                    partition[q] = partition[q].apply(generalize_country)
                elif q == "Education":
                    partition[q] = partition[q].apply(generalize_education)
                else:
                    min_val, max_val = partition[q].min(), partition[q].max()
                    partition[q] = partition[q].astype(str).apply(lambda x: f'{min_val}~{max_val}' if min_val != max_val else str(min_val))
            generalized_partitions.append(partition)
        self.partitions = generalized_partitions

    @staticmethod
    def find_median(values):
        """
        Finds the median of a given list of values.
        """
        sorted_vals = sorted(values, key=lambda x: float(x))
        mid = len(sorted_vals) // 2
        return (float(sorted_vals[mid]) + float(sorted_vals[mid - 1])) / 2 if len(sorted_vals) % 2 == 0 else float(sorted_vals[mid])

    def write_on_file(self, path):
        """
        Writes the anonymized dataset to a CSV file.
        """
        pd.concat(self.partitions).to_csv(path, index=False)

    def get_normalized_avg_equivalence_class_size(self):
        """
        Measures how well partitioning approaches the best case.
        """
        return (len(self.df) / len(self.partitions)) / self.k

    def calculate_information_loss(self):
        """
        Calculates the information loss as a percentage.
        """
        total_loss = 0
        for partition in self.partitions:
            for q in self.qi:
                if pd.api.types.is_numeric_dtype(partition[q]):
                    min_val, max_val = partition[q].min(), partition[q].max()
                    total_loss += (max_val - min_val) / (self.df[q].max() - self.df[q].min())
                else:
                    total_loss += len(partition[q].unique()) / len(self.df[q].unique())
        return (total_loss / (len(self.qi) * len(self.partitions))) * 100

def generalize_country(country):
    continent_mapping = {
        'Africa': ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'],
        'Asia': ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia', 'China', 'Cyprus', 'Georgia', 'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal', 'North Korea', 'Oman', 'Pakistan', 'Palestine', 'Philippines', 'Qatar', 'Saudi Arabia', 'Singapore', 'South Korea', 'Sri Lanka', 'Syria', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen'],
        'Europe': ['Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Georgia', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kazakhstan', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'Ukraine', 'United Kingdom', 'Vatican City'],
        'North America': ['Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic', 'El Salvador', 'Grenada', 'Guatemala', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Trinidad and Tobago', 'United States'],
        'Oceania': ['Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'New Zealand', 'Palau', 'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu'],
        'South America': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela']
    }
    for continent, countries in continent_mapping.items():
        if country in countries:
            return continent
    return 'Unknown'

def generalize_education(education):
    mapping = {
        "Middle School": "Secondary Education",
        "High School": "Secondary Education",
        "Associate Degree": "Higher Education",
        "Bachelor": "Higher Education",
        "Master": "Advanced Education",
        "PhD": "Advanced Education"
    }
    return mapping.get(education, education)

def hash_column(value):
    """Helper function to hash a single value"""
    if pd.isna(value):  # Handle NaN values
        return value
    hash_function = hashlib.sha256()
    hash_function.update(str(value).encode())  # Convert to string before hashing
    return hash_function.hexdigest()  # Return hashed value as hex

def tokenize_dataset(dataset: pd.DataFrame, columns: list):
    """
    Hashes specified columns in a pandas DataFrame.
    
    :param dataset: pandas DataFrame
    :param columns: List of column names to hash
    :return: DataFrame with hashed values in specified columns
    """
    tokenized_dataset = dataset.copy()  # Avoid modifying the original DataFrame
    
    for col in columns:
        if col in tokenized_dataset.columns:
            tokenized_dataset[col] = tokenized_dataset[col].apply(hash_column)

    return tokenized_dataset

# Example Usage
def apply_mondrian(df, k, qi, columns_to_tokenize, decoding_dict=None, output_path='anonymized_data.csv'):
    start_time = time.time()
    df = tokenize_dataset(df, columns_to_tokenize)  # Tokenize explicit identifiers
    mondrian = Mondrian(k, df, qi, decoding_dict)
    mondrian.anonymize_aux()
    execution_time = time.time() - start_time
    mondrian.write_on_file(output_path)
    nac = mondrian.get_normalized_avg_equivalence_class_size()
    info_loss = mondrian.calculate_information_loss()
    return pd.concat(mondrian.partitions), nac, info_loss, execution_time

# Load dataset
df = pd.read_csv('./data/synthetic_data_1000.csv')  # Replace with actual dataset path
qi_attributes = ['Age', 'ZipCode']  # Replace with actual quasi-identifiers
k_values = [3, 5, 10, 20, 50, 100]  # Specify the k-anonymity values to test
columns_to_tokenize = ['Name']  # Specify the columns to tokenize

nac_values = []
info_loss_values = []
execution_time_values = []

for k in k_values:
    anonymized_df, nac, info_loss, execution_time = apply_mondrian(df, k, qi_attributes, columns_to_tokenize)
    print(anonymized_df.head())
    print(f'k-Anonymity: {k}')
    print(f"Normalized Average Equivalence Class Size (NAC): {nac}")
    print(f"Information Loss (%): {info_loss}")
    print(f"Execution Time (seconds): {execution_time}")
    
    nac_values.append(nac)
    info_loss_values.append(info_loss)
    execution_time_values.append(execution_time)

# Plotting NAC vs k
plt.figure(figsize=(12, 8))
plt.plot(k_values, nac_values, marker='o', color='tab:blue', label='NAC')
plt.xlabel('k')
plt.ylabel('NAC')
plt.title('NAC vs k')
plt.grid()
plt.legend()
plt.savefig('data/nac_plot.jpg')
plt.show()

# Plotting Information Loss vs k
plt.figure(figsize=(12, 8))
plt.plot(k_values, info_loss_values, marker='o', color='tab:red', label='Information Loss (%)')
plt.xlabel('k')
plt.ylabel('Information Loss (%)')
plt.title('Information Loss vs k')
plt.grid()
plt.legend()
plt.savefig('data/info_loss_plot.jpg')
plt.show()

# Plotting Execution Time vs k
plt.figure(figsize=(12, 8))
plt.plot(k_values, execution_time_values, marker='o', color='tab:green', label='Execution Time (seconds)')
plt.xlabel('k')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs k')
plt.grid()
plt.legend()
plt.savefig('data/execution_time_plot.jpg')
plt.show()
