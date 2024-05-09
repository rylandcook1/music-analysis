import csv
import random

# Define the input and output file paths
input_file = 'test_train.csv'
output_file = 'shuffled_output.csv'

# Read the CSV file into a list of rows
with open(input_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    rows = list(csv_reader)

# Shuffle the rows
random.shuffle(rows)

# Write the shuffled rows to a new CSV file
with open(output_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(rows)

print("CSV file shuffled successfully.")