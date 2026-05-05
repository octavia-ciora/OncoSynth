# Adapted from TabDiff:
# https://github.com/MinkaiXu/TabDiff
#
# Modifications were made to integrate the code with the OncoSynth
# generation pipeline and experiment workflow.
#
# Original TabDiff copyright:
# Copyright 2024 Minkai Xu
#
# The original TabDiff code is distributed under the MIT License.

import numpy as np
import pandas as pd
import os
import sys
import json
import argparse

TYPE_TRANSFORM = {"float", np.float32, "str", str, "int", int}

parser = argparse.ArgumentParser(description="process dataset")

# General configs
parser.add_argument("--info_file", type=str, default=None, help="Path to info file.")

parser.add_argument(
    "--save_dir", type=str, default=None, help="Output dir for processed data."
)

args = parser.parse_args()


def get_column_name_mapping(
    data_df, num_col_idx, cat_col_idx, target_col_idx, column_names=None
):

    if not column_names:
        column_names = np.array(data_df.columns.tolist())

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):

        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1

    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k

    idx_name_mapping = {}

    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def train_val_test_split(data_df, cat_columns, num_train=0, num_test=0):
    total_num = data_df.shape[0]
    idx = np.arange(total_num)

    seed = 1234

    while True:
        np.random.seed(seed)
        np.random.shuffle(idx)

        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]

        train_df = data_df.loc[train_idx]
        test_df = data_df.loc[test_idx]

        flag = 0
        for i in cat_columns:
            if len(set(train_df[i])) != len(set(data_df[i])):
                flag = 1
                break

        if flag == 0:
            break
        else:
            seed += 1

    return train_df, test_df, seed


def process_data(info_file, save_dir):

    with open(info_file, "r") as f:
        info = json.load(f)

    name = info["name"]
    data_path = info["data_path"]
    if info["file_type"] == "csv":
        data_df = pd.read_csv(data_path, header=info["header"])

    elif info["file_type"] == "xls":
        data_df = pd.read_excel(data_path, sheet_name="Data", header=1)
        data_df = data_df.drop("ID", axis=1)

    num_data = data_df.shape[0]

    column_names = (
        info["column_names"] if info["column_names"] else data_df.columns.tolist()
    )

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(
        data_df, num_col_idx, cat_col_idx, target_col_idx, column_names
    )

    has_val = bool(info["val_path"])
    val_df = pd.DataFrame(columns=data_df.columns).astype(
        data_df.dtypes
    )  # by default (val_path is not provided), set val_Df to be empty
    if info["test_path"]:

        # if testing data is given
        test_path = info["test_path"]

        test_df = pd.read_csv(test_path, header=info["header"])

        if has_val:  # currently you cannot have a val path without a test path
            val_path = info["val_path"]
            val_df = pd.read_csv(val_path, header=info["header"])

        train_df = data_df

    else:
        # Train/ Test Split, 90% Training (50% for dcr eval exclusively), 10% Testing (Validation set will be selected from Training set)
        num_train = int(num_data * 0.9)
        num_test = num_data - num_train

        train_df, test_df, seed = train_val_test_split(
            data_df, cat_columns, num_train, num_test
        )

    complete_df = pd.concat([train_df, test_df, val_df], axis=0)
    name_idx_mapping = {val: key for key, val in idx_name_mapping.items()}
    int_columns = []
    int_col_idx = []
    int_col_idx_wrt_num = []
    for i, col_idx in enumerate(num_col_idx):
        col = column_names[col_idx]
        col_data = complete_df.iloc[:, col_idx]
        is_int = (col_data % 1 == 0).all()
        if is_int:
            int_columns.append(col)
            int_col_idx.append(name_idx_mapping[col])
            int_col_idx_wrt_num.append(i)
    info["int_col_idx"] = int_col_idx
    info["int_columns"] = int_columns
    info["int_col_idx_wrt_num"] = int_col_idx_wrt_num

    train_df.columns = range(len(train_df.columns))
    test_df.columns = range(len(test_df.columns))
    val_df.columns = range(len(val_df.columns))

    print(name, train_df.shape, val_df.shape, test_df.shape, data_df.shape)

    col_info = {}

    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info[col_idx]["type"] = "numerical"
        col_info[col_idx]["max"] = float(train_df[col_idx].max())
        col_info[col_idx]["min"] = float(train_df[col_idx].min())

    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info[col_idx]["type"] = "categorical"
        col_info[col_idx]["categorizes"] = list(set(train_df[col_idx]))

    for col_idx in target_col_idx:
        if info["task_type"] == "regression":
            col_info[col_idx] = {}
            col_info[col_idx]["type"] = "numerical"
            col_info[col_idx]["max"] = float(train_df[col_idx].max())
            col_info[col_idx]["min"] = float(train_df[col_idx].min())
        else:
            col_info[col_idx] = {}
            col_info[col_idx]["type"] = "categorical"
            col_info[col_idx]["categorizes"] = list(set(train_df[col_idx]))

    info["column_info"] = col_info

    train_df.rename(columns=idx_name_mapping, inplace=True)
    test_df.rename(columns=idx_name_mapping, inplace=True)
    val_df.rename(columns=idx_name_mapping, inplace=True)

    for col in num_columns:
        if (train_df[col] == " ?").sum() > 0:
            print(col)
            import pdb

            pdb.set_trace()
        if (train_df[col] == "?").sum() > 0:
            print(col)
            import pdb

            pdb.set_trace()
        train_df.loc[train_df[col] == "?", col] = np.nan
    for col in cat_columns:
        train_df.loc[train_df[col] == "?", col] = "nan"
    for col in num_columns:
        if (test_df[col] == " ?").sum() > 0:
            print(col)
            import pdb

            pdb.set_trace()
        if (test_df[col] == "?").sum() > 0:
            print(col)
            import pdb

            pdb.set_trace()
        test_df.loc[test_df[col] == "?", col] = np.nan
    for col in cat_columns:
        test_df.loc[test_df[col] == "?", col] = "nan"
    for col in num_columns:
        val_df.loc[val_df[col] == "?", col] = np.nan
    for col in cat_columns:
        val_df.loc[val_df[col] == "?", col] = "nan"

    if train_df.isna().any().any():
        print("Training data contains nan in the numerical cols")
        import pdb

        pdb.set_trace()

    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy()
    y_train = train_df[target_columns].to_numpy()

    X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
    X_cat_test = test_df[cat_columns].to_numpy()
    y_test = test_df[target_columns].to_numpy()

    X_num_val = val_df[num_columns].to_numpy().astype(np.float32)
    X_cat_val = val_df[cat_columns].to_numpy()
    y_val = val_df[target_columns].to_numpy()

    # create save_dir if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(f"{save_dir}/X_num_train.npy", X_num_train)
    np.savetxt(f"{save_dir}/X_num_train.csv", X_num_train, delimiter=",")
    np.save(f"{save_dir}/X_cat_train.npy", X_cat_train)
    np.savetxt(f"{save_dir}/X_cat_train.csv", X_cat_train, delimiter=",")
    np.save(f"{save_dir}/y_train.npy", y_train)
    np.savetxt(f"{save_dir}/y_train.csv", y_train, delimiter=",")

    np.save(f"{save_dir}/X_num_test.npy", X_num_test)
    np.savetxt(f"{save_dir}/X_num_test.csv", X_num_test, delimiter=",")
    np.save(f"{save_dir}/X_cat_test.npy", X_cat_test)
    np.savetxt(f"{save_dir}/X_cat_test.csv", X_cat_test, delimiter=",")
    np.save(f"{save_dir}/y_test.npy", y_test)
    np.savetxt(f"{save_dir}/y_test.csv", y_test, delimiter=",")

    if has_val:
        np.save(f"{save_dir}/X_num_val.npy", X_num_val)
        np.savetxt(f"{save_dir}/X_num_val.csv", X_num_val, delimiter=",")
        np.save(f"{save_dir}/X_cat_val.npy", X_cat_val)
        np.savetxt(f"{save_dir}/X_cat_val.csv", X_cat_val, delimiter=",")
        np.save(f"{save_dir}/y_val.npy", y_val)
        np.savetxt(f"{save_dir}/y_val.csv", y_val, delimiter=",")

    train_df[num_columns] = train_df[num_columns].astype(np.float32)
    test_df[num_columns] = test_df[num_columns].astype(np.float32)
    val_df[num_columns] = val_df[num_columns].astype(np.float32)

    train_df.to_csv(f"{save_dir}/train.csv", index=False)
    test_df.to_csv(f"{save_dir}/test.csv", index=False)
    if has_val:
        val_df.to_csv(f"{save_dir}/val.csv", index=False)

    # HARDCODED
    # if not os.path.exists(f"synthetic/{name}"):
    #     os.makedirs(f"synthetic/{name}")

    # train_df.to_csv(f"synthetic/{name}/real.csv", index=False)
    # test_df.to_csv(f"synthetic/{name}/test.csv", index=False)

    # if has_val:
    #     val_df.to_csv(f"synthetic/{name}/val.csv", index=False)

    print("Numerical", X_num_train.shape)
    print("Categorical", X_cat_train.shape)

    info["column_names"] = column_names
    info["train_num"] = train_df.shape[0]
    info["test_num"] = test_df.shape[0]
    info["val_num"] = val_df.shape[0]

    info["idx_mapping"] = idx_mapping
    info["inverse_idx_mapping"] = inverse_idx_mapping
    info["idx_name_mapping"] = idx_name_mapping

    metadata = {"columns": {}}
    task_type = info["task_type"]
    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    for i in num_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "numerical"
        metadata["columns"][i]["computer_representation"] = "Float"

    for i in cat_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "categorical"

    if task_type == "regression":

        for i in target_col_idx:
            metadata["columns"][i] = {}
            metadata["columns"][i]["sdtype"] = "numerical"
            metadata["columns"][i]["computer_representation"] = "Float"

    else:
        for i in target_col_idx:
            metadata["columns"][i] = {}
            metadata["columns"][i]["sdtype"] = "categorical"

    info["metadata"] = metadata

    with open(f"{save_dir}/info.json", "w") as file:
        json.dump(info, file, indent=4)

    print(f"Processing and saving {name} successfully!")

    print(name)
    print("Total", info["train_num"] + info["test_num"])
    print("Train", info["train_num"])
    print("Val", info["val_num"])
    print("Test", info["test_num"])
    if info["task_type"] == "regression":
        num = len(info["num_col_idx"] + info["target_col_idx"])
        cat = len(info["cat_col_idx"])
    else:
        cat = len(info["cat_col_idx"] + info["target_col_idx"])
        num = len(info["num_col_idx"])
    print("Num", num)
    print("Int", len(info["int_col_idx"]))
    print("Cat", cat)


if __name__ == "__main__":

    process_data(
        info_file=args.info_file,
        save_dir=args.save_dir,
    )
