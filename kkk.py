import pandas as pd
import numpy as np
import pycountry_convert as pc
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import linkage, fcluster

class Multi:
    def _init_(self, k: int, df: pd.DataFrame, qi: list):
        """
        :param k: The k-anonymity parameter
        :param df: The dataframe to anonymize
        :param qi: List of quasi-identifier columns
        """
        self.k = k
        self.df = df.copy()  # Work on a copy of the dataframe
        self.qi = qi  # Quasi-identifiers
        self.partitions = []
        
        # Define generalization hierarchy for categorical variables
        self.generalization_hierarchy = {
            'Education': {
                'High School': 'Secondary',
                'Bachelor': 'Undergraduate',
                'Master': 'Postgraduate',
                'PhD': 'Postgraduate'
            },
            'Nationality': self.generalize_nationality,  # Function for continent-based generalization
            'ZipCode': lambda x: str(x)[:2] + "XXX"  # Generalize to first two digits
        }

    def generalize_nationality(self, country_name):
        """ Convert a country name to its continent """
        try:
            country_code = pc.country_name_to_country_alpha2(country_name)
            continent_code = pc.country_alpha2_to_continent_code(country_code)

            continent_map = {
                'NA': 'North America',
                'SA': 'South America',
                'EU': 'Europe',
                'AF': 'Africa',
                'AS': 'Asia',
                'OC': 'Oceania'
            }
            return continent_map.get(continent_code, 'Unknown')
        except:
            return 'Unknown'  # If conversion fails

    def relaxed(self):
        """ Perform multidimensional relaxed anonymization (Keep all data) """
        print("Performing relaxed anonymization...")

        # Convert categorical QIs to numbers for clustering
        df_encoded = self.df.copy()
        label_encoders = {}
        
        for col in self.qi:
            if df_encoded[col].dtype == 'O':  # If categorical
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                label_encoders[col] = le

        # Perform hierarchical clustering
        qi_matrix = df_encoded[self.qi].values  
        Z = linkage(qi_matrix, method='ward')  
        cluster_labels = fcluster(Z, self.k, criterion='maxclust')  

        # Assign cluster labels
        self.df['Cluster'] = cluster_labels

        # Anonymize within clusters
        for cluster_id, cluster_group in self.df.groupby('Cluster'):
            for col in self.qi:
                if col in self.generalization_hierarchy:
                    rule = self.generalization_hierarchy[col]
                    if callable(rule):  # If generalization is a function (e.g., ZipCode, Nationality)
                        self.df.loc[self.df['Cluster'] == cluster_id, col] = cluster_group[col].apply(rule)
                    else:  # Dictionary-based generalization (Education)
                        most_common = cluster_group[col].mode()[0]
                        generalized_value = rule.get(most_common, most_common)
                        self.df.loc[self.df['Cluster'] == cluster_id, col] = generalized_value
                else:  # Numeric columns
                    # Handling Age binning based on quantiles
                    bin_labels = self.create_bins(cluster_group[col])

                    # Apply binning
                    self.df.loc[self.df['Cluster'] == cluster_id, col] = pd.cut(
                        cluster_group[col], bins=len(bin_labels), labels=bin_labels, include_lowest=True
                    )

        self.df.drop(columns=['Cluster'], inplace=True)  # Remove cluster labels
        print("Relaxed anonymization completed.")
        return self.df

    def greedy(self):
        """ Perform greedy multidimensional k-anonymization (Keep all data) """
        print("Performing greedy anonymization...")
        self._recursive_partition(self.df, self.qi)
        return self.df  # Keep all records anonymized

    def _recursive_partition(self, df, qi):
        """ Recursive function to partition the data and anonymize it in-place """
        if len(df) < self.k:
            return  # Stop partitioning if not enough data

        # Choose the best dimension for splitting
        best_col = self._find_best_split_column(df, qi)
        if not best_col:
            return  # No suitable column found

        # Split at median
        median_value = df[best_col].median()
        # For numeric, generalize to bins or ranges
        if df[best_col].dtype != 'O':  # Numeric
            bin_labels = self.create_bins(df[best_col])
            df[best_col] = pd.cut(df[best_col], bins=len(bin_labels), labels=bin_labels, include_lowest=True).fillna('Unknown')
        else:
            df[best_col] = df[best_col].mode()[0]

    def create_bins(self, column_data):
        """ Create meaningful bins based on quantiles """
        quantiles = np.percentile(column_data, np.linspace(0, 100, self.k + 1))
        bin_labels = [f"{int(quantiles[i])} - {int(quantiles[i+1])}" for i in range(self.k)]
        return bin_labels

    def _find_best_split_column(self, df, qi):
        """ Find the best column to split based on highest variance """
        variances = {col: df[col].var() for col in qi if df[col].dtype != 'O'}  # Numeric only
        return max(variances, key=variances.get, default=None)

# Example Usage
file_path = 'synthetic_data.csv'
df = pd.read_csv(file_path)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

df['ID'] = label_encoder.fit_transform(df['ID'])
df['Name'] = label_encoder.fit_transform(df['Name'])

# Define quasi-identifiers
qi_list = ['Age', 'Education', 'Nationality', 'ZipCode']

print("Original DataFrame:")
# Create anonymizer
anonymizer = Multi(k=2, df=df, qi=qi_list)

# Apply relaxed anonymization
relaxed_df = anonymizer.relaxed()
print(relaxed_df.head())

# Apply greedy anonymization
greedy_df = anonymizer.greedy()
print(greedy_df.head())