import numpy as np
import pandas as pd


def outlier_handler(data, dtype='numpy'):
    if dtype == 'numpy':
        data = pd.DataFrame(data)
    elif dtype != 'pandas':
        raise TypeError('No supported data type, please change data type to numpy or pandas!')

    data = data.replace(np.inf, np.nan)
    for feature in data:
        top = data[feature].mean() + 2 * data[feature].std()
        bottom = data[feature].mean() - 2 * data[feature].std()
        rep_max = data[data[feature] < top][feature].max()
        data.loc[data[feature] > top, feature] = rep_max
        rep_min = data[data[feature] > bottom][feature].min()
        data.loc[data[feature] < bottom, feature] = rep_min
    return data


def miss_value_handler(data, handler='drop', dtype='numpy'):
    if dtype == 'numpy':
        data = pd.DataFrame(data)
    elif dtype != 'pandas':
        raise ValueError('No supported data type, please change data type to numpy or pandas!')

    data = data.replace(np.inf, np.nan)
    data.isnull().sum()

    if handler == 'drop':
        data = data.dropna()
    elif handler == 'mean':
        for feature in data:
            data[feature] = data[feature].fillna(data[feature].mean())
    elif handler == 'interpolate':
        for feature in data:
            data[feature] = data[feature].fillna(data[feature].interpolate())
    else:
        for feature in data:
            data[feature] = data[feature].fillna(handler)

    if dtype == 'numpy':
        data = np.array(data)
    return data
