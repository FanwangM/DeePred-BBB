#!/usr/bin/env python3

"""
BBB Permeability prediction using a Deep Neural Network
as published in  https://doi.org/10.3389/fnins.2022.858126

It calculates descriptors using PaDEL, and uses them to
predict BBB permeability with a trained DNN model.

Compounds' SMILES must be in .smi file in the same folder.
Training data (data.csv), model (DeePredmodel.h5), and PaDEL
folder must be in the same folder, too.

Edited by Dr. Freddy Bernal
Modified by Dr. Fanwang Meng

"""

import os
import argparse

import pandas as pd
import numpy as np
import subprocess
from sklearn.preprocessing import StandardScaler
from keras.models import load_model


# Create function to get predictions using trained model
def bbb(folder, input_bbb: pd.DataFrame) -> np.ndarray:
    """Function to predict BBB permeability for a set
    set of samples with PaDEL features.

    Args:
        input_bbb (pd.DataFrame): PaDEL Features for compounds

    Returns:
        np.ndarray: BBB class (1 permeable; 0 non-permeable)
    """
    # Load training data and scale it
    bbb_data = pd.read_csv(os.path.join(folder, "data.csv"), header=None)
    scaler = StandardScaler()
    bbb_data = scaler.fit_transform(bbb_data)
    # Transform user data to numpy to avoid conflict with names
    bbb_user_input = scaler.transform(input_bbb.to_numpy())
    # Load model
    loaded_model = load_model(os.path.join(folder, "DeePredmodel.h5"))
    print("Model loaded")
    # Get predictions for user input
    prob = loaded_model.predict(bbb_user_input)[0, 0]
    pred_label = loaded_model.predict(bbb_user_input).round()
    pred_label = pred_label[:,0].astype(int)

    return prob, pred_label


# Create main function to run descriptor calculation and predictions
def run_prediction(folder: str, out_fname: str = None) -> None:
    """Function to calculate descriptors (using PaDEL) and to generate
    predictions of BBB permeability for a set of compounds (SMILES).

    Args:
        folder (str): Folder to search for ".smi" file (multiple structures)
        out_fname (str): Name of the output file to store predictions.

    Returns:
        predicted values for BBB permeability as a dataframe.
    """
    # Define command for PaDEL
    padel_cmd = [
        'java', '-jar',
        os.path.join(folder, 'PaDEL-Descriptor/PaDEL-Descriptor.jar'),
        '-descriptortypes',
        os.path.join(folder, 'PaDEL-Descriptor/descriptors.xml'),
        '-dir', folder, '-file', folder + '/PaDEL_features.csv',
        '-2d', '-fingerprints', '-removesalt', '-detectaromaticity',
        '-standardizenitro']
    # Calculate features
    subprocess.call(padel_cmd)
    print("Features calculated")
    # Create Dataframe for calculated features
    input_bbb =pd.read_csv(folder + "/PaDEL_features.csv")
    # Store name of each sample
    names = input_bbb['Name'].copy()
    # Remove names
    input_bbb = input_bbb.drop(['Name'], axis=1)
    # Run predictions
    prob, pred_label = bbb(folder, input_bbb)
    # Create Dataframe with results
    res = pd.DataFrame(names)
    res['Predicted_class'] = pred_label
    res["predicted_probability"] = prob
    # Save results to csv
    if out_fname:
        res.to_csv(out_fname, index=False)

    return res


# Run script
if __name__ == "__main__":
    # Define current directory
    # if len(sys.argv) == 2:
    #     path = sys.argv[1]
    # else:
    #     path = os.getcwd()
    # # Verify existence of file with SMILES
    # exists = [fname for fname in os.listdir(path) if fname.endswith(".smi")]
    # if exists:
    #     # Get predictions
    #     run_prediction(path)
    # else:
    #     raise FileNotFoundError("Input File NOT found")

    argparser = argparse.ArgumentParser(description='BBB Permeability prediction using a Deep Neural Network')
    argparser.add_argument('--wdir', type=str, help='Folder to search for ".smi" file (multiple structures)')
    argparser.add_argument('--out_fname', type=str,default=None, help='Name of the output file to store predictions.')
    argparser.add_argument('--verbose', action='store_true', help='Print verbose output')

    args = argparser.parse_args()
    res = run_prediction(args.wdir, args.out_fname)

    if args.verbose:
        print(res)
