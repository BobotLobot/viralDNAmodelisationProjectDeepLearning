import csv
from argparse import ArgumentParser
import matplotlib.pyplot as plt

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_csv_file", required=True)
    parser.add_argument("-o", "--output_file", default="evaluation_fig.png")
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

    fig, ax = plt.subplots()
    ax.plot(pred_radii, corr_radii, ".-")
    ax.plot(pred_pitches, corr_pitches, ".-")
    fig.savefig(args.output_file)
    
main()
