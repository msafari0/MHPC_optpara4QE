import os
import sys
import tempfile
sys. path.append("../src")
import json
import pandas as pd
import numpy as np
import argparse
import read_outputs
import clean_data
import glob
import ann_model as ann
from collections.abc import MutableMapping
from ase.data import atomic_numbers
from matplotlib import pyplot as plt
import seaborn as sns
import GPy
import itertools
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def extract_raw_parameters_from_qe_output(base_path, machine, algo_name="davidson"):
    combined_data = []
    print(f"Searching for QE output files under: {base_path}")

    for root, dirs, files in os.walk(base_path):
        for f in files:
            file_path = os.path.join(root, f)
            try:
                with open(file_path, 'r', errors='ignore') as fr:
                    head = fr.read(5000)
                    if "This program is part of the open-source Quantum ESPRESSO suite" in head:
                        out_json_path = os.path.join(root, "data.json")
                        data = read_outputs.create_json(
                            folder=root,
                            outname=out_json_path,
                            platform=machine,
                            algoname=algo_name
                        )
                        combined_data.extend(data)
                        break  # Only need to detect one QE file per folder
            except Exception:
                continue

    unified_json = os.path.join(base_path, "data-leo-tot.json")
    print(f"Writing unified dataset to: {unified_json}")
    with open(unified_json, "w") as fw:
        json.dump(combined_data, fw, indent=2)

    print(f"Extraction completed. Total entries: {len(combined_data)}")
    return combined_data, unified_json


def train_save_model(raw_json_path=None, data=None, model_path=None, epochs=80):
    print(" Cleaning data...")

    if data is not None:
        # Write data to temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(data, tmp, indent=2)
            tmp_path = tmp.name

        # Clean the temporary JSON file
        df_leo = clean_data.clean_data(tmp_path)

        # Remove temporary file
        os.remove(tmp_path)
    elif raw_json_path is not None:
        df_leo = clean_data.clean_data(raw_json_path)
    else:
        raise ValueError("Please provide either 'data' or 'raw_json_path'.")

    print(f"Dataset size: {len(df_leo)} samples")

    elements = list(clean_data.atomic_numbers.keys())
    drop_cols = [
        'time_per_call', 'n_calls', 'normalized_time_per_call', 'convergence',
        'Al1', 'Cl1', 'Co1', 'Cr1', 'F1', 'H1', 'H2', 'Ir1', 'La1', 'N1',
        'Na1', 'Nb1', 'Ni1', 'O1', 'O2', 'Re1', 'S1', 'Si1', 'Ti1', 'V1'
    ] + elements
    
    # Drop unwanted columns
    dfx = df_leo.drop(columns=[c for c in drop_cols if c in df_leo.columns])

    # Keep only numeric columns
    dfx = dfx.select_dtypes(include=['number']).astype('float')
  #  dfx = df_leo.drop(columns=[c for c in drop_cols if c in df_leo.columns]).astype('float')
    dfy = df_leo['normalized_time_per_call'].astype('float')

    print(f"Training model with {len(dfx.columns)} input features...")

    model = ann.TimePerCall(
        activation='swish',
        l1=1e-5,
        l2=1e-4,
        lr=0.0005,
        loss='mae',
        nvars=len(dfx.columns)
    )

    ind_tr, ind_val = ann.train_test_indices(dfx)
    X_tr, Y_tr = dfx.loc[ind_tr], dfy.loc[ind_tr]
    X_val, Y_val = dfx.loc[ind_val], dfy.loc[ind_val]

    model.train_normed(
        X_tr, Y_tr,
        epochs=epochs,
        validation_split=0.2,
        plot=1,
        save_path="training_plot.png"
    )

    train_loss, val_loss = model.evaluate_losses(
        model.normalize_x(X_tr),
        model.normalize_y(Y_tr),
        model.normalize_x(X_val),
        model.normalize_y(Y_val)
    )
    print(f"Final losses → Train: {train_loss:.4f}, Val: {val_loss:.4f}")
    # Determine model save path properly
    if model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "model.keras")
    else:
        if not os.path.isabs(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, model_path)

    model.save(model_path)
    print(f" Model saved to {model_path}")

    #script_dir = os.path.dirname(os.path.abspath(__file__))  # directory of pipeline.py
    #model_path = os.path.join(script_dir, "model_leo.keras")

    return model


def full_pipeline(base_path, machine, raw_json_path=None, algo_name="davidson", epochs=80, reuse_json=True, model_path=None):
    """
    Full QE data extraction + cleaning + model training pipeline.
    """
    if raw_json_path is not None:
        unified_json = os.path.join(base_path, raw_json_path)
    else:
        unified_json = os.path.join(base_path, "data-leo-tot.json")

    if reuse_json and os.path.exists(unified_json):
        print(f"Reusing existing JSON: {unified_json}")
        combined_data = None
    else:
        print("Extracting QE parameters...")
        combined_data, unified_json = extract_raw_parameters_from_qe_output(
            base_path, machine, algo_name
        )

    # Determine where to save model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(script_dir, "model_full_pipeline.keras")

    model = train_save_model(
        raw_json_path=unified_json if combined_data is None else None,
        data=combined_data,
        model_path=model_path,
        epochs=epochs
    )

    print(f" Full pipeline completed successfully. Model saved at: {model_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full QE data extraction and ML training pipeline")
    parser.add_argument("--base_path", type=str, default="../data-leo", help="Root folder with QE outputs")
    parser.add_argument("--machine", type=str, default="Leonardo-booster", help="Machine/platform name")
    parser.add_argument("--algo_name", type=str, default="davidson", help="Algorithm name")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--reuse_json", action="store_true", help="Reuse existing unified JSON if present")
    parser.add_argument("--json_path", type=str, default=None,
                        help="Path to an existing unified JSON to skip extraction")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Custom path/name for saving trained model")  # <── NEW ARG HERE

    args = parser.parse_args()

    # Determine which data to use
    """  if args.json_path is not None:
        print(f" Using provided JSON file: {args.json_path}")
        model = train_save_model(
            raw_json_path=args.json_path,
            model_path=args.model_path,
            epochs=args.epochs
        )
    else:"""
    model = full_pipeline(
            base_path=args.base_path,
            machine=args.machine,
            algo_name=args.algo_name,
            epochs=args.epochs,
            reuse_json=args.reuse_json,
            model_path=args.model_path
    )

"""if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full QE data extraction and ML training pipeline")
    parser.add_argument("--base_path", type=str, default="../data-leo", help="Root folder with QE outputs")
    parser.add_argument("--machine", type=str, default="Leonardo-booster", help="Machine/platform name")
    parser.add_argument("--algo_name", type=str, default="davidson", help="Algorithm name")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--reuse_json", action="store_true", help="Reuse existing unified JSON if present")
    parser.add_argument("--json_path", type=str, default=None,
                        help="Path to an existing unified JSON to skip extraction")

    args = parser.parse_args()

    # Determine which data to use
    if args.json_path is not None:
        # Use provided JSON file
        print(f" Using provided JSON file: {args.json_path}")
        model = train_save_model(raw_json_path=args.json_path, epochs=args.epochs)
    else:
        # Run full pipeline (with optional reuse_json)
        model = full_pipeline(
            base_path=args.base_path,
            machine=args.machine,
            algo_name=args.algo_name,
            epochs=args.epochs,
            reuse_json=args.reuse_json
        )"""
