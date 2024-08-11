import argparse
import logging
import os.path
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError
from tqdm import tqdm

from motrack.utils import file_system

logger = logging.getLogger('FilterMotInference')


def filter_dataframe_by_column(df1: pd.DataFrame, column1: str, df2: pd.DataFrame, column2: str) -> pd.DataFrame:
    """
    Removes rows from the first DataFrame (df1) if the value in a specified column (column1)
    does not exist in a specific column (column2) of the second DataFrame (df2).

    Args:
        df1 (pd.DataFrame): The first DataFrame to be filtered.
        column1 (str): The column name in df1 whose values need to exist in df2's column2.
        df2 (pd.DataFrame): The second DataFrame used as a reference for filtering df1.
        column2 (str): The column name in df2 that contains values to be matched against df1's column1.

    Returns:
        pd.DataFrame: A filtered version of df1 containing only rows where the value in column1 exists in column2 of df2.

    Example:
        >>> df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> df2 = pd.DataFrame({'X': [3, 4, 5], 'Y': [6, 7, 8]})
        >>> filter_dataframe_by_column(df1, 'A', df2, 'X')
            A  B
        2  3  6
    """
    unique_values_in_df2 = df2[column2].unique()  # Extract unique values to optimize the isin operation
    filtered_df1 = df1[df1[column1].isin(unique_values_in_df2)]  # Filter df1 based on values in df2
    return filtered_df1


def main(args: argparse.Namespace) -> None:
    input_path: str = args.inference_input
    output_path: str = args.inference_output
    dataset_path: str = args.dataset_path

    file_names = file_system.listdir(input_path, regex_filter='(.*?)[.]txt')
    Path(output_path).mkdir(parents=True, exist_ok=True)
    for file_name in tqdm(file_names, unit='scene', desc='Filtering un-annotated frames'):
        file_input_path = os.path.join(input_path, file_name)
        file_output_path = os.path.join(output_path, file_name)
        file_ann_path = os.path.join(dataset_path, file_name.replace('.txt', ''), 'gt', 'gt.txt')

        try:
            df_input = pd.read_csv(file_input_path, header=None)
        except EmptyDataError:
            logger.warning(f'Inference file "{file_input_path}" is empty! Creating empty output file...')
            open(file_output_path, 'w', encoding='utf-8').close()
            continue

        df_input.columns = ['frame_id', 'object_id', 'xmin', 'ymin', 'w', 'h', 'confidence', '-1', '-1', '-1']

        df_ann = pd.read_csv(file_ann_path, header=None)
        df_ann = df_ann.iloc[:, :6]
        df_ann.columns = ['frame_id', 'object_id', 'xmin', 'ymin', 'w', 'h']

        df_output = filter_dataframe_by_column(df_input, 'frame_id', df_ann, 'frame_id')
        df_output.to_csv(file_output_path, header=False, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Removes from inference all frames that are not annotated.')
    parser.add_argument('--inference-input', type=str, required=True, help='Path where the inference output is stored.')
    parser.add_argument('--inference-output', type=str, required=True, help='Path where the filtered inference should be stored')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path where the dataset is stored.')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
