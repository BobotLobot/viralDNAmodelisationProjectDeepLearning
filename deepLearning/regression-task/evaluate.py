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

    plt.plot(pred_radii, corr_radii, marker='.', linestyle='')
    plt.xlabel("predicted radius")
    plt.ylabel("correct radius")
    plt.xticks(np.linspace(np.min(pred_radii), np.max(pred_radii), NUMBER_OF_TICKS))
    plt.savefig(args.output_file_prefix+"-radii.png")

    plt.plot(pred_pitches, corr_pitches, marker='.', linestyle='')
    plt.xlabel("predicted pitch")
    plt.ylabel("correct pitch")
    plt.xticks(np.linspace(np.min(pred_pitches), np.max(pred_pitches), NUMBER_OF_TICKS))
    plt.savefig(args.output_file_prefix+"-pitches.png")
    
main()
