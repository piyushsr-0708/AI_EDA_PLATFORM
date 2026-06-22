import pandas as pd


def clean_dataset(df,profile):
    clean_report={}
    
    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates()
    clean_report['duplicates_removed'] = duplicates_removed

    converted_num = []
    for column in profile['numeric_like_text']:
        df[column] = df[column].astype(str).str.replace(',','',regex = False).str.replace('%','',regex = False).str.replace('$','',regex = False).str.replace('+','',regex = False)
        df[column] = pd.to_numeric(df[column],errors='coerce')
        converted_num.append(column)
    clean_report['numeric_converted'] = converted_num

    converted_date = []
    for column in profile['date_candidates']:
        df[column] = pd.to_datetime(df[column], errors = 'coerce')
        converted_date.append(column)
    clean_report['date_converted'] = converted_date

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            mode_value = df[column].mode()
            if len(mode_value)>0:
                df[column] = df[column].fillna(mode_value[0])
    clean_report['remaining_missing'] = int(df.isnull().sum().sum())
    clean_report['rows_after_cleaning'] = len(df)

    return df,clean_report
