import os
from collections import defaultdict
from pathlib import Path
import pandas as pd
from glob import glob
import torch

# define the results root
parent_directory = Path(__file__).resolve().parents[1]
root_directory = parent_directory / 'results'

# define the list of indicator dimensions, methods, and queries
indicator_dimensions = ['2D', '3D', '4D']
methods = ['AABox', 'OBox', 'Sphere', 'AAElli', 'OElli', 'kDOP', 'oursKDOP', 'oursNeural']
queries = ['Point', 'Ray', 'Plane', 'Box']

# create a defaultdict with default value as an empty list
results = defaultdict(list)

# iterate through the indicator dimensions, queries, and methods
for indicator_dimension in indicator_dimensions:
    for query in queries:
        path = os.path.join(root_directory, query.lower(), indicator_dimension)
        csv_files = glob(os.path.join(path, '*', '*.csv'))

        for csv_file in csv_files:
            df = pd.read_csv(csv_file, header=0, index_col=0)
            object_name = os.path.basename(os.path.dirname(csv_file))

            for method in methods:
                result = round(df.loc[method, 'false positives'] * 100 / df.loc[method, 'total samples'], 1)
                results[(method, f'{indicator_dimension} {query}')].append(result)

# calculate the average per indicator dimension, query, and method
for key, value in results.items():
    tensor = torch.tensor(value, dtype=torch.float)
    results[key] = round(torch.mean(tensor).item(), 1)

# define the headers for rows and columns of Table 1
row_headers = methods
column_headers = [f'{d} {q}' for d in indicator_dimensions for q in queries]

# make Table 1 and save to a csv file in the results directory
df = pd.DataFrame(index=row_headers, columns=column_headers)
for row_header in row_headers:
    for col_header in column_headers:
        key = (row_header, col_header)
        if key in results:
            df.at[row_header, col_header] = results[key]
df.to_csv(root_directory / 'table1.csv')
