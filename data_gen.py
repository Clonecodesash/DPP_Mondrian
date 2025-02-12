from faker import Faker
import pandas as pd
import random

# Initialize the Faker object
fake = Faker()
Faker.seed(42)

# Define the number of records you want to generate
num_records = 10

# Generate unique four-digit IDs
unique_ids = random.sample(range(1000, 10000), num_records)

# Initialize an empty list to store the data
data = []

# Generate the data
for i in range(num_records):
    gender = fake.random_element(elements=('Male', 'Female'))
    if gender == 'Male':
        name = fake.name_male()
    else:
        name = fake.name_female()
    
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    education_weights = [0.5, 0.5, 0.3, 0.1]
    education = random.choices(education_levels, weights=education_weights, k=1)[0]
    
    record = {
        'ID': unique_ids[i],
        'Name': name,
        'Age': fake.random_int(min=20, max=60),
        'Gender': gender,
        'ZipCode': fake.random_int(min=15500, max=19999),
        'Education': education,
        'Nationality': fake.country(),
        'Salary': fake.random_int(min=30000, max=120000)
    }
    data.append(record)

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('synthetic_data.csv', index=False)

print("Synthetic data generated and saved to 'synthetic_data.csv'")
