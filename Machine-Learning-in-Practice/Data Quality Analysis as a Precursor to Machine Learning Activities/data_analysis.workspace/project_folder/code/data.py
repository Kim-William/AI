import pandas as pd

# Load the CSV file into a DataFrame (modify the file path as needed)
data = pd.read_excel('Data/SIS_Faculty-List.xlsx')

# Set the key column for identification
key_column = 'ID'

# Define a function to replace values
def replace_value(x):
    x = str(x).lower()
    if 'master' in x or 'mba' in x or 'ma' in x :
        return 'Masters'
    elif 'phd' in x or 'ph.d' in x or 'ph. d' in x or 'doctor' in x:
        return 'Doctorate'
    elif 'bachelor' in x:
        return 'Bachelors'
    return x

# Apply value replacement to specific columns
data['Highest\nQualification\nLevel'] = data['Highest\nQualification\nLevel'].apply(replace_value)
data['Highest Qualification'] = data['Highest Qualification'].apply(replace_value)

# Detect rows with duplicate ID values
duplicate_id_rows = data[data.duplicated(subset=key_column, keep=False)]

# Detect rows with NaN ID values
nan_id_rows = data[data[key_column].isnull()]

# Count the number of duplicate ID values and NaN ID values
duplicate_id_count = duplicate_id_rows[key_column].value_counts()
nan_id_count = nan_id_rows.shape[0]

# Count the total number of rows in the dataset
total_rows = data.shape[0]

# Print information about duplicate and NaN ID values
if not duplicate_id_count.empty or nan_id_count > 0:
    error_row_count = 0
    print(f"Number of duplicate {key_column} values and NaN {key_column} values:")
    
    if not duplicate_id_count.empty:
        print("Number of duplicate ID values:")
        for id_value, count in duplicate_id_count.items():
            error_row_count += count
            print(f"ID: {id_value}, Duplicate Count: {count}")
    
    if nan_id_count > 0:
        error_row_count += nan_id_count
        print(f"Number of NaN ID values: {nan_id_count}")
        
    print(f"Total rows: {total_rows}")
    print(f"Bad rows: {error_row_count}")
    print(f"Valid rows: {((total_rows - error_row_count) / total_rows) * 100:.2f}%")
else:
    print("No rows with duplicate ID or NaN ID.")
    print(f"Total rows: {total_rows}")

# Create a dictionary to store common values among duplicate rows
common_values = {}

# Find common values for each column among duplicate rows
for col in duplicate_id_rows.columns:
    common_values[col] = set()
    for idx1, row1 in duplicate_id_rows.iterrows():
        for idx2, row2 in duplicate_id_rows.iterrows():
            if idx1 != idx2:
                if row1[col] == row2[col]:
                    common_values[col].add(row1[col])

# Create a dictionary to store error values for each column
error_col_value = {}

# Check for columns with common values and determine the most common value
for col, common in common_values.items():
    max_duplicate_rate = {}
    if len(common) > 0:
        for value in common:
            max_duplicate_rate[value] = (duplicate_id_rows[col] == value).sum()

# Find the most common value
max_key = max(max_duplicate_rate, key=max_duplicate_rate.get)
max_value = max_duplicate_rate[max_key]

# Check if the most common value is present in the majority of rows
count = (data[col] == max_key).sum()
if count == max_value:
    error_col_value[col] = max_key


# Print columns with error values
for key, value in error_col_value.items():
    print(f'{key}: {value}')

# Filter out rows with error values in the key column
data = data[(data[key_column] != value) & (~data[key_column].isnull())]

# Define the output file path
output_file_path = 'Data\Result.csv'

# Save the modified data to a CSV file
data.to_csv(output_file_path, index=False)

# Print a message indicating the completion of data processing
print(f"Rows with ID 0 or NaN {key_column} values have been removed and saved to '{output_file_path}'.")

# Check for data types in columns and recommend type standardization if necessary
for col in data:
    if col == 'LWD' or col == 'ID' or col == 'Name':
        continue
    df_dup = data[col].drop_duplicates()
    
    if df_dup.size > 2:
        print(f"Below are the data types for the '{col}' column. Check if type standardization is possible.")
        print(df_dup)
        print("===============================\r\n")
        
# ....................................... DATA CLEANING .......................................

# Handle missing data
data['Location'].fillna('Unknown', inplace=True)  # Replace missing Location values with 'Unknown'
data.dropna(subset=['Reports To'], inplace=True)  # Remove rows with missing Reports To values

# Value standardization
data['Location'] = data['Location'].str.strip().str.title()  # Remove leading/trailing spaces and capitalize Location values
data['Title'] = data['Title'].str.strip()  # Remove leading/trailing spaces from Title values
data['Join\nDate'] = pd.to_datetime(data['Join\nDate'], errors='coerce')  # Convert Join Date to datetime format, handle invalid entries

# Standardize Highest Qualification values
data['Highest Qualification'] = data['Highest Qualification'].apply(replace_value)

# Save cleaned data
output_file_path = 'Data/Result_cleaned.xlsx'
data.to_excel(output_file_path, index=False)

# Convert the Result_cleaned.xlsx file to a CSV file
data = pd.read_excel('Data/Result_cleaned.xlsx')
output_file_path = 'reports/Result_cleaned.csv'
data.to_csv(output_file_path, index=False)


# Print a message indicating the completion of data processing
print(f"Cleaned data has been saved to '{output_file_path}'.")

# Generate a report of the cleaned data with their rows and columns 
report = data.describe(include='all').transpose()
report.insert(0, 'Column', report.index)
report.reset_index(drop=True, inplace=True)
report.to_csv('reports/Report.csv', index=False)

print("========================== \n")





# Print a message indicating the completion of data processing
print(f"Report has been saved to 'reports/Report.csv'.")