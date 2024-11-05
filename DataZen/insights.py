import pandas as pd
from scipy import stats
import numpy as np
import warnings

def get_columns(df):
    columns = list(df.columns)
    return columns

def numerical_info(df):
    numerical_corr = []
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(numeric_cols) > 1:

        corr_matrix = df[numeric_cols].corr()
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    direction = "positive" if corr > 0 else "negative"
                    numerical_corr.append(f"Strong {direction} correlation of {corr:.2f} exists between {numeric_cols[i]} and {numeric_cols[j]}.")
                elif abs(corr) > 0.5:
                    direction = "positive" if corr > 0 else "negative"
                    numerical_corr.append(f"Moderate {direction} correlation of {corr:.2f} exists between {numeric_cols[i]} and {numeric_cols[j]}.")
        # corr_matrix = df[numeric_cols].corr()
        # for i in range(len(numeric_cols)):
        #     for j in range(i+1, len(numeric_cols)):
        #         if abs(corr_matrix.iloc[i, j]) > 0.5:
        #             print(f"numerical relation found between {numeric_cols[i]} and {numeric_cols[j]}")
        #             numerical_corr.append(numeric_cols[i])
        #             numerical_corr.append(numeric_cols[j])

    return (numerical_corr)


def categorical_numerical_info(df):
    cat_num_relationships = []
    cat_num_insights = []
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            groups = [group[num_col].values for name, group in df.groupby(cat_col)]
            # print(groups)
            # Filter out groups with less than 2 samples
            valid_groups = [group for group in groups if len(group) >= 2]
            # print(valid_groups)
        
            if len(valid_groups) >= 2:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)
                        f_statistic, p_value = stats.f_oneway(*valid_groups)
                    # print("p_value :", p_value)
                    if not np.isnan(p_value) and p_value < 0.05:
                        # print("entered loop")
                        group_means = df.groupby(cat_col)[num_col].mean()
                        max_category = group_means.idxmax()
                        min_category = group_means.idxmin()
                        cat_num_insights.append(f"Significant relationship exists between {cat_col} and {num_col} with value = {p_value}. "
                                        f"Highest average {num_col} : {max_category}, Lowest average {num_col} : {min_category}.")
                        cat_num_relationships.append(f"Significant relationship between {cat_col} and {num_col}")
                    # else:
                    #     print("not entered")
                except Exception as e:
                    pass
                    print(f"Error in ANOVA for {cat_col} and {num_col}: {str(e)}")
            else:
                print(f"Skipping ANOVA for {cat_col} and {num_col} due to insufficient group sizes")

    # print("cat_ num Relationships found:")
    # for relationship in cat_num_relationships:
    #     print(relationship)
    # print("cat_num_insights")
    # for relation in cat_num_insights:
    #     print(relation)
    # print(cat_num_corr)

    return cat_num_insights


def time_series_info(df):
    time_series_relation = []
    time_series_insights = []
    date_cols = df.select_dtypes(include=['datetime64']).columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(date_cols) > 0:
        for date_col in date_cols:
            for num_col in numeric_cols:
                time_series_relation.append(f"Potential time series relationship between {date_col} and {num_col}")

        # Grouping by Month
        for num_col in numeric_cols:
            if pd.api.types.is_datetime64_any_dtype(df['Month']):
                df['Month'] = df['Month'].dt.month
            
            if df['Month'].dtype != 'int64':  # Handle non-integer months
                try:
                    df['Month'] = df['Month'].astype(int)
                except:
                    raise ValueError("Month column could not be converted to integer.")
            group_means = df.groupby('Month')[num_col].mean()
            max_month = group_means.idxmax()
            min_month = group_means.idxmin()
            time_series_insights.append(f"Significant relationship between Month and {num_col}. "
                                        f"Highest {num_col} : {max_month}, Lowest {num_col} : {min_month}.")
    
    # Grouping by Year
        for num_col in numeric_cols:
            if pd.api.types.is_datetime64_any_dtype(df['Year']):
                df['Year'] = df['Year'].dt.year
            group_means = df.groupby('Year')[num_col].mean()
            max_year = group_means.idxmax()
            min_year = group_means.idxmin()
            time_series_insights.append(f"Significant relationship between Year and {num_col}. "
                                        f"Highest {num_col} : {max_year}, Lowest {num_col} : {min_year}.")
    # print("time_series Relationships found:")
    # for relationship in time_series_relation:
    #     print(relationship)
    # print("change in time series : ")
    # for relation in time_series_insights:
    #     print(relation)
    
    return time_series_insights


# Categorical data distribution
def imbalanced_distribution(df):
    imbalanced_relation = []
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for cat_col in categorical_cols:
        value_counts = df[cat_col].value_counts(normalize=True)
        if (value_counts > 0.8).any():  # If any category represents more than 80% of the data
            imbalanced_relation.append(f"Highly imbalanced distribution in {cat_col}")
    return imbalanced_relation

# Detect potential outliers
# def outliers(df):
#     outlier_relation = []
#     numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
#     for num_col in numeric_cols:
#         Q1 = df[num_col].quantile(0.25)
#         Q3 = df[num_col].quantile(0.75)
#         IQR = Q3 - Q1
#         if ((df[num_col] < (Q1 - 1.5 * IQR)) | (df[num_col] > (Q3 + 1.5 * IQR))).any():
#             outlier_relation.append(f"Potential outliers detected in {num_col}")
#     return outlier_relation
def outliers(df):
    outlier_relation = []
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for num_col in numeric_cols:
        Q1 = df[num_col].quantile(0.25)
        Q3 = df[num_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Boolean mask to check for outliers
        outliers_mask = (df[num_col] < lower_bound) | (df[num_col] > upper_bound)
        outliers_count = outliers_mask.sum()
        
        if outliers_count > 0:
            outlier_relation.append(f"Potential outliers detected in {num_col}: {outliers_count} outliers")
    
    return outlier_relation



# dataset = "../StudentsPerformance.csv"
# df = pd.read_csv(dataset)

# if df.get('Year') is not None:
#     df['Year'] = pd.to_datetime(df['Year'], format='%Y')
# if df.get('Month') is not None:
#     df['Month'] = pd.to_datetime(df['Month'], format='%m')
# if df.get('Day') is not None:
#     df['Day'] = pd.to_datetime(df['Day'], format='%d')

# num_info_results = numerical_info(df)
# print("Numerical info :")
# print("\n")
# for a in num_info_results:
#     print(a)
# print("\n")

# cat_num_info_results = categorical_numerical_info(df)
# print("categorical and numerical info :")
# print("\n")
# for a in cat_num_info_results:
#     print(a)
# print("\n")

# time_series_results = time_series_info(df)
# print("Time series :")
# print("\n")
# for a in time_series_results:
#     print(a)
# print("\n")

# imbalanced_distribution_results = imbalanced_distribution(df)
# print("Imbalanced distribution :")
# print("\n")
# for a in imbalanced_distribution_results:
#     print(a)
# print("\n")

# outliers_result = outliers(df)
# print("Outliers :")
# print("\n")
# for a in outliers_result:
#     print(a)
# print("\n")
# print(visualization_goal_gen(num_info_results, cat_num_info_results, time_series_results))
# print(time_series(dataset))

