import pandas as pd
from faker import Faker
import random

# Initialize the Faker object
fake = Faker()
Faker.seed(45)

# Define the sizes of datasets you want to generate
dataset_sizes = [10,50, 100,500, 1000,5000, 10000]

# Define possible values for categorical attributes
genders = ['Male', 'Female']
education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
education_weights = [0.5, 0.3, 0.15, 0.05]
nationalities = [fake.country() for _ in range(100)]  # Generate a list of 100 random countries

# Function to generate synthetic data
def generate_data(num_records):
    # Generate unique digit IDs
    unique_ids = random.sample(range(1000, 100000), num_records)
    
    # Initialize an empty list to store the data
    data = []
    
    # Generate the data
    for i in range(num_records):
        gender = fake.random_element(elements=genders)
        if gender == 'Male':
            name = fake.name_male()
        else:
            name = fake.name_female()
        
        education = random.choices(education_levels, weights=education_weights, k=1)[0]
        
        record = {
            'ID': unique_ids[i],
            'Name': name,
            'Age': fake.random_int(min=20, max=60),
            'Gender': gender,
            'ZipCode': fake.random_int(min=15500, max=19999),
            'Education': education,
            'Nationality': fake.random_element(elements=nationalities),
            'Salary': fake.random_int(min=30000, max=120000)
        }
        data.append(record)
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    return df

# Generate datasets of different sizes
for size in dataset_sizes:
    df = generate_data(size)
    filename = f'synthetic_data_{size}.csv'
    df.to_csv(filename, index=False)
    print(f"Synthetic data of size {size} generated and saved to '{filename}'")
