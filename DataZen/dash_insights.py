import pandas as pd
from scipy import stats
import numpy as np
import warnings

def numerical_insights(df):
    num_insights = []
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    direction = "positive" if corr > 0 else "negative"
                    num_insights.append(f"Strong {direction} correlation of {corr*100:.2f} % exists between {numeric_cols[i]} and {numeric_cols[j]}.")
                elif abs(corr) > 0.5:
                    direction = "positive" if corr > 0 else "negative"
                    num_insights.append(f"Moderate {direction} correlation of {corr*100:.2f} exists between {numeric_cols[i]} and {numeric_cols[j]}.")

    return (num_insights)


def categorical_numerical_info(df):
    cat_num_insights = {}
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    numeric_cols = numeric_df.loc[:, numeric_df.nunique() > 10].columns
    # categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtype in [ 'object', 'category','int64']]
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
                    # print("\n categorical ",cat_col,"Numerical ",num_col)
                    # if not np.isnan(p_value) and p_value <= 0.05:
                        # print("entered loop")
                        group_means = df.groupby(cat_col)[num_col].mean()
                        max_category = group_means.idxmax()
                        min_category = group_means.idxmin()
                        cat_num_insights[(cat_col, num_col)] = f"Highest average {num_col} : {max_category} and Lowest average {num_col} : {min_category}"
                        # cat_num_insights.append(f"Significant relationship exists between {cat_col} and {num_col} with value = {p_value}. "
                        #                 f"Highest average {num_col} : {max_category}, Lowest average {num_col} : {min_category}.")
                    # else:
                    #     print("not entered")
                except Exception as e:
                    pass
                    print(f"Error in ANOVA for {cat_col} and {num_col}: {str(e)}")
            else:
                print(f"Skipping ANOVA for {cat_col} and {num_col} due to insufficient group sizes")

    return cat_num_insights


def time_series_info(df):

    time_series_insights = []
    date_cols = df.select_dtypes(include=['datetime64']).columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(date_cols) > 0:
        # for date_col in date_cols:
        #     for num_col in numeric_cols:
        #         time_series_relation.append(f"Potential time series relationship between {date_col} and {num_col}")

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
    
    return time_series_insights


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


def imbalanced_distribution(df):
    imbalanced_relation = []
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for cat_col in categorical_cols:
        value_counts = df[cat_col].value_counts(normalize=True)
        if (value_counts > 0.8).any():  # If any category represents more than 80% of the data
            imbalanced_relation.append(f"Highly imbalanced distribution in {cat_col}")
    return imbalanced_relation


def time_series_info(df):
    # time_series_relation = []
    time_series_insights = {}
    date_cols = df.select_dtypes(include=['datetime64']).columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(date_cols) > 0:
        # for date_col in date_cols:
        #     for num_col in numeric_cols:
                # time_series_relation.append(f"Potential time series relationship between {date_col} and {num_col}")

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
            time_series_insights[('Month', num_col)] = f"Highest average {num_col} : {max_month}, Lowest average {num_col} : {min_month}."
            # time_series_insights.append(f"Significant relationship between Month and {num_col}."
            #                             f"Highest {num_col} : {max_month}, Lowest {num_col} : {min_month}.")
    
    # Grouping by Year
        for num_col in numeric_cols:
            if pd.api.types.is_datetime64_any_dtype(df['Year']):
                df['Year'] = df['Year'].dt.year
            group_means = df.groupby('Year')[num_col].mean()
            max_year = group_means.idxmax()
            min_year = group_means.idxmin()
            time_series_insights[('Year', num_col)] = f"Highest average {num_col} : {max_year}, Lowest average {num_col} : {min_year}."
            # time_series_insights.append(f"Significant relationship between Year and {num_col}. "
            #                             f"Highest {num_col} : {max_year}, Lowest {num_col} : {min_year}.")
    # print("time_series Relationships found:")
    # for relationship in time_series_relation:
    #     print(relationship)
    # print("change in time series : ")
    # for relation in time_series_insights:
    #     print(relation)
    
    return time_series_insights


def outliers(df):
    outlier_insights = {}
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
            outlier_insights[(num_col)] = f"Number of ouliers in {num_col} : {outliers_count}"
            # outlier_relation.append(f"Potential outliers detected in {num_col}: {outliers_count} outliers")
    
    return outlier_insights