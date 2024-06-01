# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
# Load your stock market data into a DataFrame
# Replace 'your_data.csv' with the actual filename or data source
df = pd.read_csv('^NSEI.csv')
df['Volume'] = df['Volume'].astype(float)
df['Date'] = pd.to_datetime(df['Date'], format ='%Y-%m-%d')
# Step 1: Inspect the Data
# Display the first few rows of the dataset
#print("First Few Rows of Data:")
#print(df.head())

# Check data types and non-null counts
#print("\nData Info:")
#print(df.info())

# Step 2: Descriptive Statistics
# Generate summary statistics for numeric columns
print("\nSummary Statistics:")
print(df.describe())

# Step 3: Data Visualization
# Example: Plot histograms for numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns
for column in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

'''# Step 4: Time Series Analysis (if applicable)
# Example: Plot stock price over time
numeric_columns = df.select_dtypes(include=['number']).columns
for column in numeric_columns:
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df[column])
    plt.title(column + ' Over Time')
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.xticks(rotation=45)
    plt.show()'''

# Step 6: Handling Missing Data (if applicable)
# Example: Check for missing values and display columns with missing data
#missing_data = df.isnull().sum()
#print("\nMissing Data:")
#print(missing_data[missing_data > 0])

# Step 7: Correlation Analysis (if applicable)
# Example: Calculate and visualize correlations
correlation_matrix = df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Step 8: Outlier Identification (if applicable)
# Example: Identify and visualize potential outliers
numeric_columns = df.select_dtypes(include=['number']).columns
for column in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[column])
    plt.title('Box Plot of ' + column)
    plt.xlabel(column)
    plt.show()

# Step 10: Documentation
# Keep detailed notes in a separate document or notebook.
