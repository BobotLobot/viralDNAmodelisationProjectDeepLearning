import csv
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

NUMBER_OF_TICKS = 10

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_csv_file", required=True)
    parser.add_argument("-o", "--output_file_prefix", default="evaluation-fig")
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
    plt.savefig(args.output_file_prefix+"-radii.png")

    fig, ax = plt.subplots()
    ax.plot(pred_pitches, corr_pitches, marker='.', linestyle='')
    ax.set_xlabel("predicted pitch")
    ax.set_ylabel("correct pitch")
    x_ticks = np.linspace(np.min(pred_pitches), np.max(pred_pitches), NUMBER_OF_TICKS)
    ax.set_xticklabels(
    [f"{label:.2f}" for label in x_ticks]
    )
    plt.savefig(args.output_file_prefix+"-pitches.png")
    
main()
