import csv
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

NUMBER_OF_TICKS = 10

def get_mean_pred_per_corr(corr_vars : list, pred_vars : list) -> dict:
    """
        returns a dict with corrct values as keys and model predictions for each key in a list as values.
    """
    corrs_dict = dict()
    for i in range(len(corr_vars)): # dict key is correct value, values are all model outputs for the value
        if corr_vars[i] in corrs_dict:
            corrs_dict[corr_vars[i]].append(pred_vars[i])
        else:
            corrs_dict[corr_vars[i]] = [pred_vars[i]]

    mean_preds = dict()
    for corr in corrs_dict:
        mean_preds[corr] = np.mean(corrs_dict[corr])

    print("mean_preds:", mean_preds)
    return mean_preds

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_csv_file", required=True)
    parser.add_argument("-o", "--output_file_prefix", default="evaluation")
    args = parser.parse_args()
    
    pred_radii = []
    pred_pitches = []
    corr_radii = []
    corr_pitches = []
    with open(args.input_csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_radii.append(float(row["pred_radius"]))
            pred_pitches.append(float(row["pred_pitch"]))
            corr_pitches.append(float(row["correct_pitch"]))
            corr_radii.append(float(row["correct_radius"]))

    fig, ax = plt.subplots()
    ax.plot(pred_radii, corr_radii, marker='.', linestyle='')
    ax.set_xlabel("predicted radius")
    ax.set_ylabel("correct radius")
    x_ticks = np.linspace(np.min(pred_radii), np.max(pred_radii), NUMBER_OF_TICKS)
    ax.set_xticklabels(
    [f"{label:.2f}" for label in x_ticks]
    )
    mean_preds_dict = get_mean_pred_per_corr(corr_radii, pred_radii)
    print("moved mean_preds_dict:", mean_preds_dict)
    ax.plot([mean_preds_dict[key] for key in mean_preds_dict.keys()], mean_preds_dict.keys(), 'x')
    plt.savefig(args.output_file_prefix+"-fig-radii.png")

    fig, ax = plt.subplots()
    ax.plot(pred_pitches, corr_pitches, marker='.', linestyle='')
    ax.set_xlabel("predicted pitch")
    ax.set_ylabel("correct pitch")
    x_ticks = np.linspace(np.min(pred_pitches), np.max(pred_pitches), NUMBER_OF_TICKS)
    ax.set_xticklabels(
    [f"{label:.2f}" for label in x_ticks]
    )
    mean_preds_dict = get_mean_pred_per_corr(corr_pitches, pred_pitches)
    ax.plot([mean_preds_dict[key] for key in mean_preds_dict.keys()], mean_preds_dict.keys(), 'x')
    plt.savefig(args.output_file_prefix+"-fig-pitches.png")
    
main()
