from pathlib import Path

import mnist
import numpy as np
import pandas as pd

from src.PARAMATERS import data_external_dir

data_external_dir = Path(data_external_dir)
##############
# Regression #
##############


def load_white_wine(normalise=True):
    df = pd.read_csv(data_external_dir / 'wine' /
                     'winequality-white.csv', sep=';')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values.astype(np.float64)
    if normalise:
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        y -= y.mean()
        y /= y.std()

    return X, y


def load_red_wine(normalise=True):
    df = pd.read_csv(data_external_dir / 'wine' /
                     'winequality-red.csv', sep=';')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values.astype(np.float64)
    if normalise:
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        y -= y.mean()
        y /= y.std()

    return X, y


def load_student_performance_math(normalise=True):
    student_math_df = pd.read_csv(
        data_external_dir / 'student_performance' / 'student-mat.csv', sep=';')

    # preprocess data, turn categorical into dummies
    student_math_df = pd.get_dummies(student_math_df)

    if normalise:
        # Normalise these columns
        norm_cols = ['age',
                     'Medu',
                     'Fedu',
                     'traveltime',
                     'studytime',
                     'failures',
                     'famrel',
                     'freetime',
                     'goout',
                     'Dalc',
                     'Walc',
                     'health',
                     'absences']
        student_math_df[norm_cols] = student_math_df[norm_cols].apply(
            lambda x: (x - x.mean()) / x.std())
    output_col = 'G3'
    keep_cols = list(set(student_math_df.columns) - set(output_col))

    X = student_math_df[keep_cols].values
    y = student_math_df[output_col].values.astype(np.float64)

    if normalise:
        y -= y.mean()
        y /= y.std()

    return X, y


def load_student_performance_port(normalise=True):
    student_port_df = pd.read_csv(
        data_external_dir / 'student_performance' / 'student-por.csv', sep=';')

    # preprocess data, turn categorical into dummies
    student_port_df = pd.get_dummies(student_port_df)

    if normalise:
        # Normalise these columns
        norm_cols = ['age',
                     'Medu',
                     'Fedu',
                     'traveltime',
                     'studytime',
                     'failures',
                     'famrel',
                     'freetime',
                     'goout',
                     'Dalc',
                     'Walc',
                     'health',
                     'absences']
        student_port_df[norm_cols] = student_port_df[norm_cols].apply(
            lambda x: (x - x.mean()) / x.std())
    output_col = 'G3'
    keep_cols = list(set(student_port_df.columns) - set(output_col))

    X = student_port_df[keep_cols].values
    y = student_port_df[output_col].values.astype(np.float64)

    if normalise:
        y -= y.mean()
        y /= y.std()

    return X, y


def load_bike_sharing_day(normalise=True):
    """Load daily bike_sharing dataset and preprocess it"""
    bike_sharing_day_df = pd.read_csv(
        data_external_dir / 'bike_sharing' / 'day.csv')

    # Only keep these columns as regressors
    keep_cols = ['holiday', 'workingday', 'temp', 'atemp', 'hum', 'windspeed']
    if normalise:
        # Normalise these columns
        norm_cols = ['temp', 'atemp', 'hum', 'windspeed']
        bike_sharing_day_df[norm_cols] = bike_sharing_day_df[norm_cols].apply(
            lambda x: (x - x.mean()) / x.std())
    # We predict bike sharing count 'cnt'
    output_col = 'cnt'

    X = bike_sharing_day_df[keep_cols].values
    y = bike_sharing_day_df[output_col].values.astype(np.float64)

    if normalise:
        y -= y.mean()
        y /= y.std()

    return X, y


def load_concrete(normalise=True):
    """Load concrete dataset and preprocess it"""
    df = pd.read_csv(data_external_dir / 'concrete' / 'Concrete_Data.csv')

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(np.float64)

    if normalise:
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        y -= y.mean()
        y /= y.std()

    return X, y


###########################
# Classification : Binary #
###########################


def load_cencus_income(normalise=True):
    """Load census income dataset and preprocess it"""
    adult_test_df = pd.read_csv(
        data_external_dir / 'census_income' / 'adult.test', na_values=[' ?'], index_col=False)
    adult_df = pd.read_csv(data_external_dir / 'census_income' /
                           'adult.data', na_values=[' ?'], index_col=False)

    adult_full_df = (pd.concat([adult_test_df, adult_df]).replace(to_replace=' ?', value=np.nan)
                     .dropna())

    # Only keep these columns as regressors
    keep_cols = ['age', 'fnlwgt', 'education-num',
                 'capital-gain', 'capital-loss', 'hours-per-week']
    if normalise:
        # Normalise these columns
        norm_cols = ['age', 'fnlwgt', 'education-num',
                     'capital-gain', 'capital-loss', 'hours-per-week']
        adult_full_df[norm_cols] = adult_full_df[norm_cols].apply(
            lambda x: (x - x.mean()) / x.std())
    # We predict bike sharing count 'cnt'
    # from string to binary
    adult_full_df['rich'] = adult_full_df['rich'].apply(
        lambda x: int('>' in x))
    output_col = 'rich'

    X = adult_full_df[keep_cols].values
    y = adult_full_df[output_col].values

    return X, y


#########################
# Classification: Multi #
#########################


def load_mnist(normalise=True):
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    train_images = train_images.reshape(-1, 28 * 28).astype(float)
    test_images = test_images.reshape(-1, 28 * 28).astype(float)

    X = np.vstack([train_images, test_images])
    y = np.concatenate([train_labels, test_labels])

    if normalise:
        X -= X.mean(axis=0)
        X /= (X.std(axis=0) + 1e-6)  # Some pixels are always 0

    return X, y


def load_yeast(normalise=True):
    # changed the downloaded file
    df = pd.read_csv(data_external_dir / 'yeast' / 'yeast.csv')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

    if normalise:
        X -= X.mean(axis=0)
        X /= X.std(axis=0)

    y = pd.get_dummies(y).values.argmax(1)

    return X, y
