import pandas as pd

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
        dim_ranges = {q: self.df[q].max() - self.df[q].min() for q in self.qi}
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
        fs = partition[dim].values
        split_val = self.find_median(fs)
        lhs_rhs = self.create_partition(partition, dim, split_val)

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
                min_val, max_val = partition[q].min(), partition[q].max()
                if q in self.decoding_dict:
                    partition[q] = partition[q].astype(str).map(lambda x: self.decoding_dict[q].get(x, x))
                else:
                    partition[q] = partition[q].astype(str).apply(lambda x: f'{min_val}~{max_val}' if min_val != max_val else str(min_val))
            generalized_partitions.append(partition)
        self.partitions = generalized_partitions

    @staticmethod
    def find_median(values):
        """
        Finds the median of a given list of values.
        """
        sorted_vals = sorted(values)
        mid = len(sorted_vals) // 2
        return (sorted_vals[mid] + sorted_vals[mid - 1]) / 2 if len(sorted_vals) % 2 == 0 else sorted_vals[mid]

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

# Example Usage
def apply_mondrian(df, k, qi, decoding_dict=None, output_path='anonymized_data.csv'):
    mondrian = Mondrian(k, df, qi, decoding_dict)
    mondrian.anonymize_aux()
    mondrian.write_on_file(output_path)
    return pd.concat(mondrian.partitions)

# Load dataset
df = pd.read_csv('synthetic_data.csv')  # Replace with actual dataset path
qi_attributes = ['Age', 'ZipCode']  # Replace with actual quasi-identifiers
k_value = 3
anonymized_df = apply_mondrian(df, k_value, qi_attributes)
print(anonymized_df.head())
