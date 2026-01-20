import sys
import glob
import os
sys. path.append("./src")
import clean_data
import read_outputs
import json
import numpy as np
import NNimaker
import pandas as pd
import ann_model as ann
from collections.abc import MutableMapping
from ase.data import atomic_numbers
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
import GPy
import itertools
import joblib

elements = list(atomic_numbers.keys())

QE_SIGNATURE = "This program is part of the open-source Quantum ESPRESSO suite"

def is_qe_output_folder(folder):
    """Check if any file in the folder looks like a Quantum ESPRESSO output."""
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if not os.path.isfile(file_path):
            continue
        try:
            with open(file_path, "r", errors="ignore") as f:
                head = f.read(5000)  # read first few KB only
                if QE_SIGNATURE in head:
                    return True
        except Exception:
            pass
    return False
    
def extract_raw_parameters_from_qe_output(base_path, machine, algo_name="davidson"):
    """
    Recursively searches for QE output folders (based on file contents),
    extracts parameters, and writes both per-folder and unified JSON files.
    """

    combined_data = []
    qe_output_folders = []

    for root, dirs, files in os.walk(base_path):
        if is_qe_output_folder(root):
            qe_output_folders.append(root)

    if not qe_output_folders:
        print(f" No QE output folders found under {base_path}")
        return []

    print(f"Found {len(qe_output_folders)} QE output folders")

    for folder_path in qe_output_folders:
        folder_name = os.path.basename(folder_path)
        out_json = os.path.join(folder_path, "data.json")

        try:
            data = read_outputs.create_json(
                folder=folder_path,
                outname=out_json,
                platform=machine,
                algoname=algo_name
            )
            combined_data.extend(data)
            print(f"Extracted from {folder_name} ({len(data)} entries)")
        except Exception as e:
            print(f"Error processing {folder_name}: {e}")
            continue

    unified_path = os.path.join(base_path, "data-leo-tot.json")
    with open(unified_path, "w") as fw:
        json.dump(combined_data, fw, indent=2)

    print(f"\nUnified JSON written to: {unified_path}")
    print(f" Total extracted entries: {len(combined_data)}")

    return combined_data


def train_save_model(raw_json_path="../data-leo/data-leo-tot.json", save_path="model_leonardo.keras"):
    """
    Cleans raw QE data, trains a neural network model, and saves it to disk.

    Parameters
    ----------
    raw_json_path : str
        Path to the combined JSON data file (output from extraction step).
    save_path : str
        Path where the trained model should be saved.
    """

    # Load and clean the data
    print(" Loading and cleaning data...")
    df_leo = clean_data.clean_data(raw_json_path)

    # Drop unnecessary columns
    drop_cols = [
        'time_per_call', 'n_calls', 'normalized_time_per_call', 'convergence',
        'Al1', 'Cl1', 'Co1', 'Cr1', 'F1', 'H1', 'H2', 'Ir1', 'La1', 'N1', 'Na1',
        'Nb1', 'Ni1', 'O1', 'O2', 'Re1', 'S1', 'Si1', 'Ti1', 'V1'
    ]

    # If you have an `elements` list elsewhere, add it dynamically
    try:
        from elements import elements  # optional, if you have this defined in another file
        drop_cols += elements
    except ImportError:
        pass

    drop_cols = [c for c in drop_cols if c in df_leo.columns]  # safety filter
    dfx = df_leo.drop(columns=drop_cols).astype(float)
    dfy = df_leo["normalized_time_per_call"].astype(float)

    print(f" Data ready: {dfx.shape[0]} samples, {dfx.shape[1]} features")

    # Create model
    print("Initializing model...")
    model = ann.TimePerCall(
        activation='swish',
        l1=1e-5,
        l2=1e-4,
        lr=0.0005,
        loss='mae',
        nvars=len(dfx.columns)
    )

    # Split train/validation data
    print("Splitting training and validation sets...")
    ind_tr, ind_val = ann.train_test_indices(dfx)
    X_tr, Y_tr = dfx.loc[ind_tr], dfy.loc[ind_tr]
    X_val, Y_val = dfx.loc[ind_val], dfy.loc[ind_val]

    # Train the model
    print(" Training model...")
    model.train_normed(
        X_tr, Y_tr,
        epochs=80,
        validation_split=0.2,
        plot=1,
        save_path="training_plot0005.png"
    )

    # Evaluate losses
    print("Evaluating performance...")
    train_loss, val_loss = model.evaluate_losses(
        model.normalize_x(X_tr), model.normalize_y(Y_tr),
        model.normalize_x(X_val), model.normalize_y(Y_val)
    )
    print(f"Final losses — Train: {train_loss:.4f}, Val: {val_loss:.4f}")

    #Save model
    model.save(save_path)
    print(f" Model saved to: {save_path}")

    return model

"""def extract_raw_parameters_from_qe_output(folders, machine)

    data = []

    for f in folders:
        data_leo.extend(read_outputs.create_json(folder=f"../data-leo/{f}/outputs/", outname= f"../data-leo/{f}/data.json", platform='Leonardo-booster', algoname='davidson'))

    #from module read_outputs.py!
    for f2 in folders2:
        data_leo.extend(read_outputs.create_json(folder=f"../data-leo/{f2}/", outname= f"../data-leo/{f2}/data.json", platform='Leonardo-booster', algoname='davidson'))

    with open("../data-leo/data-leo-tot.json", 'w') as fw:
        json.dump(data_leo, fw, indent=2)
    
    return
"""
    

df_leo = clean_data.clean_data("../data-leo/data-leo-tot.json") #This command read the raw data of all runs and add normalized column and all the necessary things.

df_leo.to_pickle("../data-leo/all_data.xz") #to save the cleaned data as all_data.xz

dfx = df_leo.drop(columns = ['time_per_call', 'n_calls', 'normalized_time_per_call', 'convergence', 'Al1', 'Cl1', 'Co1', 'Cr1', 'F1', 'H1',
       'H2', 'Ir1', 'La1', 'N1', 'Na1', 'Nb1', 'Ni1', 'O1', 'O2', 'Re1', 'S1',
       'Si1', 'Ti1', 'V1'] + elements).astype('float')
#dfx.columns
dfy = df_leo['normalized_time_per_call'].astype('float')
# dfx.arch = 10

def train_save_model()

model = ann.TimePerCall(activation='swish', l1=1e-5,
             l2=1e-4, lr=0.0005, loss='mae', nvars = len(dfx.columns))
ind_tr, ind_val = ann.train_test_indices(dfx)
X_tr = dfx.loc[ind_tr]
Y_tr = dfy.loc[ind_tr]
X_val = dfx.loc[ind_val]
Y_val = dfy.loc[ind_val]
model.train_normed(X_tr, Y_tr, epochs=80, validation_split=0.2, plot=1, save_path="training_plot0005.png")
# After training, check proper losses:
train_loss, val_loss = model.evaluate_losses(
    model.normalize_x(X_tr), 
    model.normalize_y(Y_tr),
    model.normalize_x(X_val), 
    model.normalize_y(Y_val)
)
model.save("model_leonardo.keras")

if __name__ == '__main__':
    print('Welcome Message:')
    print('\tTraining model on clusters.')
    print('\tArguments: ...')
    print('\t Author: Mandana Safari \n')
    parser = argparse.ArgumentParser(description="Extract info from pseudopotential files.")
    parser.add_argument('--file', type=str)
    args = parser.parse_args()



