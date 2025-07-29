import pandas as pd
import plotly.express as px

def impute(df: pd.DataFrame, imputed: str, imputed_on: str, stat: str) -> pd.DataFrame:
    """
    Imputes the missing values in a specified column (imputed) based on its mean across another specified column (imputed_on).
    If no mean is found within a group, the global mean is used.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data to be processed.
        imputed (str): The column in the DataFrame that contains missing values to be imputed.
        imputed_on (str): The column used to group the data and compute the mode of the 'imputed' column.
    Returns:
        pd.DataFrame: The DataFrame with imputed values.
    """
    nulls_df = df[df[imputed].isnull()]
    #A slice containing only rows with null values for imputed col

    if stat == 'mean':
        stats = df.groupby(imputed_on)[imputed].mean().fillna(df[imputed].mean()).to_dict()
        #Save the mean of 'imputed' col values for each group in the 'imputed_on' column in means dictionary
        #If no mean is found for a group, use the global mean
    elif stat == 'median':
        stats = df.groupby(imputed_on)[imputed].median().fillna(df[imputed].median()).to_dict()
    elif stat == 'mode':
        global_mode = df[imputed].value_counts().index[0]
        stats = df.groupby(imputed_on)[imputed].agg(lambda x: x.mode()[0] if not x.mode().empty else global_mode).to_dict()        

    nulls_df[imputed] = nulls_df[imputed_on].map(stats)
    #Impute using mapped mean

    df.loc[nulls_df.index, imputed] = nulls_df[imputed]
    #Fill missing values in original dataframe
    
    return df

def box_plt(df: pd.DataFrame, x: str, y=None) -> None:
    """
    Creates and displays a box plot using Plotly.
    If y not provided, the function will plot a box plot for the x-axis column alone.
    If y provided, x is the categorical variable and y is the numerical variable.

    Args:
    -df (pandas.DataFrame): The dataframe containing the data to be visualized.
    -x (str): The column name to be used for the x-axis.
    -y (str, optional): The column name to be used for the y-axis. Defaults to None.

    """
    if x != None:
        fig = px.box(df, x=x, y=y)
    else:
        fig = px.box(df, x)
    fig.show()

def date_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates meaningful features out of date columns ('date_account_created' and 'timestamp_first_active').
        Args:
            df (pd.DataFrame): The DataFrame containing the dates column to be processed.
        Returns:
            pd.DataFrame: The DataFrame with imputed values.
        """
        df['date_account_created'] = pd.to_datetime(df['date_account_created'])
        df['timestamp_first_active'] = pd.to_datetime(df['timestamp_first_active'], format="%Y-%m-%d %H:%M:%S")

        df['hour_first_active'] = df['timestamp_first_active'].dt.hour
        df['day_first_active'] = df['timestamp_first_active'].dt.day
        df['weekday_first_active'] = df['timestamp_first_active'].dt.weekday
        df['month_first_active'] = df['timestamp_first_active'].dt.month
        df['year_first_active'] = df['timestamp_first_active'].dt.year

        df['day_account_created'] = df['date_account_created'].dt.day
        df['weekday_account_created'] = df['date_account_created'].dt.weekday
        df['month_account_created'] = df['date_account_created'].dt.month
        df['year_account_created'] = df['date_account_created'].dt.year

        df = df.drop(columns=['date_account_created', 'timestamp_first_active']) 


def map_rare(df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps categories with a small value count to one "other" category to prepare data for target encoding.
        Args:
            df (pd.DataFrame): The DataFrame containing the categorical columns to be processed.
        Returns:
            pd.DataFrame: The DataFrame with mapped rare categories.
        """
        rare = {'language': ['is', 'ca', 'th', 'hu', 'el', 'fi', 'cs', 'no', 'pl', 'da', 'tr', 'nl', 'sv'],
            'affiliate_provider': ['daum', 'yandex', 'baidu', 'naver', 'email-marketing', 'gsp', 'meetup', 'facebook-open-graph', 'yahoo'],
            'first_affiliate_tracked': ['local_ops', 'marketing', 'product'],
            'signup_flow': ['10', '20', '16', '5', '15', '21', '8', '6', '1'],
            'first_browser': ['Opera', 'Silk', 'Chromium', 'BlackBerry Browser', 'Maxthon', 'IE Mobile', 'Apple Mail', 'Sogou Explorer',
            'Mobile Firefox', 'SiteKiosk', 'RockMelt', 'Iron', 'IceWeasel', 'Pale Moon', 'SeaMonkey', 'CometBird', 'Yandex.Browser', 'Camino',
            'TenFourFox', 'CoolNovo', 'wOSBrowser', 'Avant Browser', 'Opera Mini', 'Mozilla', 'TheWorld Browser', 'Flock', 'Comodo Dragon',
            'OmniWeb', 'SlimBrowser', 'Crazy Browser', 'Opera Mobile', 'Epic', 'Stainless', 'Googlebot', 'Outlook 2007', 'Arora', 'NetNewsWire',
            'PS Vita browser', 'Google Earth', 'Conkeror', 'Kindle Browser', 'Palm Pre web browser', 'IceDragon']}
        
        for key, rare_values in rare.items():
            df[key] = df[key].apply(lambda x: 'other' if x in rare_values else x)