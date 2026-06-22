import csv
from argparse import ArgumentParser
import matplotlib.pyplot as plt

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_csv_file", required=True)
    parser.add_argument("-o", "--output_file_prefix", default="evaluation_fig")
    args = parser.parse_args()
    
    pred_radii = []
    pred_pitches = []
    corr_radii = []
    corr_pitches = []
    with open(args.input_csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_radii.append(row["pred_radius"])
            pred_pitches.append(row["pred_pitch"])
            corr_pitches.append(row["correct_pitch"])
            corr_radii.append(row["correct_radius"])

    plt.plot(pred_radii, corr_radii, ".-")
    plt.xlabel("predicted radius")
    plt.ylabel("correct radius")
    plt.savefig(args.output_file_prefix+"-radii.png")

    plt.plot(pred_pitches, corr_pitches, ".-")
    plt.xlabel("predicted pitch")
    plt.ylabel("correct pitch")
    plt.savefig(args.output_file_prefix+"-pitches.png")
    
main()
