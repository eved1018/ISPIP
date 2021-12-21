# Evan Edelstein
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef


def trainningsetstats():
    df = pd.read_csv(
        "/Users/evanedelstein/Desktop/MetaDPIv2/metadpi/input/input_data_all.csv")

    df["protein"] = [x.split('_')[1] for x in df['residue']]
    df.isnull().any()
    df = df.fillna(method='ffill')

    cutoff_dict = cutoff_file_parser(
        "/Users/evanedelstein/Desktop/MetaDPIv2/metadpi/input/cutoffs.csv")
    output = pd.DataFrame()
    for i in ['predus', 'ispred', 'dockpred']:
        top = df.sort_values(by=[i], ascending=False).groupby((["protein"])).apply(
            lambda x: x.head(cutoff_dict[x.name])).index.get_level_values(1).tolist()

        df[f'{i}_bin'] = [
            1 if i in top else 0 for i in df.index.tolist()]

        fscore_mcc_per_protein = df.groupby((["protein"])).apply(
            lambda x: fscore(x, 'annotated', i))

        output[f"{i}_mcc"] = fscore_mcc_per_protein

    print(output)
    output.to_csv(
        "/Users/evanedelstein/Desktop/MetaDPIv2/metadpi/output/training_set_stats/mcc.csv")
    return


def fscore(x, annotated_col, pred) -> tuple:
    return matthews_corrcoef(x[annotated_col], x[f'{pred}_bin'])


def cutoff_file_parser(cutoff_frame) -> dict:
    cutoff_frame_df: pd.DataFrame = pd.read_csv(cutoff_frame)
    cutoff_frame_df.columns = cutoff_frame_df.columns.str.lower()
    cutoff_dict = dict(
        zip(cutoff_frame_df['protein'], cutoff_frame_df['cutoff res']))
    return cutoff_dict


trainningsetstats()
