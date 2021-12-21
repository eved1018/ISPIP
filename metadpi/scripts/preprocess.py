# Evan Edelstein
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from metadpi.scripts.containers.argscontainer import ArgsContainer


def data_preprocesss(df: pd.DataFrame) -> tuple:
    feature_cols: list = df.columns.tolist()[1:-1]
    annotated_col: str = df.columns.tolist()[-1]
    df["protein"] = [x.split('_')[1] for x in df['residue']]
    proteins: np.ndarray = df["protein"].unique()
    df.set_index('residue', inplace=True)
    df.isnull().any()
    df = df.fillna(method='ffill')
    df = df[df['annotated'] != "ERROR"]
    df["annotated"] = pd.to_numeric(df["annotated"])
    return df, feature_cols, annotated_col, proteins


def data_split_auto(df, proteins) -> tuple:
    test_set, train_set = train_test_split(proteins, test_size=0.2)
    train_frame: pd.DataFrame = df[df["protein"].isin(train_set)]
    test_frame: pd.DataFrame = df[df["protein"].isin(test_set)]
    return test_frame, train_frame


def data_split_from_file(df: pd.DataFrame, args_container: ArgsContainer) -> tuple:
    test_file = args_container.test_proteins_file
    train_file = args_container.train_proteins_file
    test_frame, lines = input_parser(test_file, df)
    train_frame, lines = input_parser(train_file, df)
    return test_frame, train_frame


def cross_validation_set_generater(cvs_path: str, df: pd.DataFrame) -> tuple:
    cvs: list = []
    train_proteins: list = []
    test_frame = None
    for file_name in os.listdir(cvs_path):
        if file_name.startswith("training"):
            file_name_path = os.path.join(cvs_path, file_name)
            train_frame, lines = input_parser(file_name_path, df)
            cvs.append(train_frame)
            train_proteins.append(lines)

        elif file_name.startswith("test"):
            file_name_path = os.path.join(cvs_path, file_name)
            test_frame, _ = input_parser(file_name_path, df)
        else:
            print("please include test and train sets")

    return test_frame, cvs, train_proteins


def input_parser(file, df) -> tuple:
    with open(file) as f:
        lines = f.read().rstrip().splitlines()
    lines = [i if i[-2] == '.' else f"{i[:-1]}.{i[-1:]}" if i[-2].isdigit(
    ) or i[-2].isalpha() else i.replace(i[-2], '.') for i in lines]
    frame: pd.DataFrame = df[df['protein'].isin(lines)]
    return frame, lines
