import pandas as pd


def profile_dataset(df):

    profile = {}

    profile["rows"] = df.shape[0]
    profile["columns"] = df.shape[1]

    profile["duplicate_rows"] = int(df.duplicated().sum())

    profile["total_missing"] = int(df.isnull().sum().sum())

    numerical_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    profile["numerical_columns"] = numerical_cols
    profile["categorical_columns"] = categorical_cols

    datetime_cols = df.select_dtypes(
        include=['datetime64[ns]']
    ).columns.tolist()

    profile["datetime_cols"] = datetime_cols

    profile["missing_by_column"] = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )

    profile["unique_values"] = {
        col: int(df[col].nunique())
        for col in df.columns
    }

    profile["column_types"] = {
        col: str(df[col].dtype)
        for col in df.columns
    }

    binary_columns = []

    for column in df.columns:
        unique_count = df[column].nunique(dropna=True)
        if unique_count == 2:
            binary_columns.append(column)
    profile['binary_columns'] = binary_columns
    
    date_candidates = []
    for column in df.select_dtypes(include=['object', 'category']).columns:
        sample = df[column].dropna().astype(str).head(50)
        if sample.empty:
            continue

        converted = pd.to_datetime(sample, errors='coerce')
        success_rate = converted.notna().sum() / len(sample)

        if success_rate < 0.8:
            converted = pd.to_datetime(
                sample,
                errors='coerce',
                dayfirst=True
            )
            success_rate = converted.notna().sum() / len(sample)

        if success_rate >= 0.8:
            date_candidates.append(column)
    profile['date_candidates'] = date_candidates

    numeric_like = []
    for column in df.select_dtypes(include = ['object', 'category']).columns:
        sample = df[column].dropna().head(50)
        if len(sample)==0:
            continue
        sample = sample.astype(str).str.replace(',','',regex = False).str.replace('%','',regex = False).str.replace('$','',regex = False).str.replace('+','',regex = False)
        converted = pd.to_numeric(sample, errors = 'coerce')
        success_rate = converted.notna().sum()/len(sample)
        if success_rate>0.9:
            numeric_like.append(column)
    profile['numeric_like_text'] = numeric_like

    low_cardinality = []
    for column in df.columns:
        c_unique = df[column].nunique(dropna=True)
        if c_unique <= 10:
            low_cardinality.append(column)
    profile['low_cardinality'] = low_cardinality

    high_cardinality = []
    for column in df.columns:
        c_unique = df[column].nunique(dropna=True)
        u_ratio = c_unique/len(df)
        if u_ratio > 0.5:
            high_cardinality.append(column)
    profile['high_cardinality'] = high_cardinality
    
    return profile