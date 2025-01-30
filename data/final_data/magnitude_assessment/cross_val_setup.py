import pandas as pd
from sklearn.model_selection import train_test_split

input_file = 'tuned_reasoning.csv'
output_dir = 'train_test_splits_reasoning'

import os
os.makedirs(output_dir, exist_ok=True)

data = pd.read_csv(input_file)

for i in range(1, 6):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=i)
    
    train_file = os.path.join(output_dir, f'train_split_{i}.csv')
    test_file = os.path.join(output_dir, f'test_split_{i}.csv')
    
    train_data.to_csv(train_file, index=False, encoding='utf-8')
    test_data.to_csv(test_file, index=False, encoding='utf-8')

    print(f"Created train-test pair {i}:")
    print(f"  Train file: {train_file}")
    print(f"  Test file: {test_file}")
