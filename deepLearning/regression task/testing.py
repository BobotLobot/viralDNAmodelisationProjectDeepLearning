import torch
import numpy as np
from trainingWithMeta import Model4
from model import MrcDataset1vMetaDataWithNoiseFile, DataLoader
import mrcfile
from torchvision.transforms.functional import get_dimensions
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os.path

PREDICTION_BATCH_SIZE = 20

def get_label(mrcdic : dict) -> long:
    if mrcdic["onlyNoise"]:
        label = 2 #score when no data present
    else:
        label = mrcdic["zEnd"]
    return label

def get_data(args : argparse.Namespace):
    print("loading dataset...")
    dataset = MrcDataset1vMetaDataWithNoiseFile(metaFile=args.meta_file,
    noiseDirectory=args.noisy_data_dir,
    noNoiseDirectory=args.not_noisy_data_dir,
    onlyNoise = args.only_noise_dir) # note: copied from other model
    return dataset

def get_metadata(dataset, num_datapoints : int) -> list:
    metadata = dataset.MrcFiles[0:num_datapoints]
    if len(metadata) != num_datapoints:
        raise ValueError("Incorrect metadata indexes used. Length of collected metadata array:", len(metadata))
    return metadata

def get_map(filename, dataset):
    # Loading data
    try:
        with mrcfile.open(filename, mode='r+') as mrc:
            mrcData = mrc.data #extraction of the data
    except (ValueError, OSError) as e:
        print(f"file {mrcfileName} is corrupted. Error : {str(e)}")
        return None
    mindata=np.min(mrcData)
    for i in range(mrcData.shape[0]):
        mrcData[i][0][0]=0.0
    fileStd=np.std(mrcData)
    fileMean=np.mean(mrcData)
    mrcData = (mrcData - fileMean) / fileStd
    mrcData = torch.from_numpy(mrcData).float()
    #label = torch.tensor(label, dtype=torch.long)
    #data augmentation
    
    mrcData = dataset.augmentation(mrcData)
    
    # Add channel dimension
    mrcData = mrcData.unsqueeze(0)
    if dataset.transform:
        mrcData = dataset.transform(mrcData)
    return mrcData

def print_test(data, model) -> None:
    trainDataloader = DataLoader(data, batch_size=2, shuffle=True)
    for i, data in enumerate(trainDataloader,0):
        inputs, labels = data
        inputs = inputs.float()
        labels = labels
        outputs = model(inputs)
        print("inputs type:", type(inputs))
        print("inputs:", get_dimensions(inputs))
        print("input len:", len(inputs)) # equal to batch size, not the dimensions of the map cube
        print("outputs:", get_dimensions(outputs))
        return # if not included, prints once per batch until all data is used

def benchmark(model, dataset, verbose, out_path, num_predictions):
    metadata = get_metadata(dataset, num_predictions)
    maps = []
    correct_radii = []
    correct_pitches = []
    for m in metadata:
        maps.append(get_map(m["filename"], dataset))
        correct_radii.append(m["radius"])
        correct_pitches.append(m["pitch"])
    maps = torch.tensor(np.array(maps))
    
    preds_to_make_tot = len(metadata)
    
    if verbose:
        print("number of predictions to make:", preds_to_make_tot)
        print("making predictions...")
    preds = []
    while True:
        preds_made = len(preds)
        preds_to_make_now = min(preds_to_make_tot - preds_made, PREDICTION_BATCH_SIZE)
        preds = preds + list(model( maps[ preds_made : preds_made + preds_to_make_now]))

        preds_made = preds_made + preds_to_make_now
        if verbose:
            print(f" {preds_made} predictions have been made")
        if preds_made == preds_to_make_tot:
            break
    
    pred_radii = []
    pred_pitches = []
    if verbose:
        print("constructing plots...")
    for pred in preds:
        pred_radii.append(pred[0])
        pred_pitches.append(pred[1])

    if verbose:
        print("writing predictions to file:", out_path)
    with open(out_path, "w") as f:
        f.write("pred_radius,pred_pitch,correct_radius,correct_pitch\n")
        for i in range(len(preds)):
            f.write(f"{pred_radii[i]},{pred_pitches[i]},{correct_radii[i]},{correct_pitches[i]}\n")

def get_args() -> argparse.Namespace:
    parser = ArgumentParser()
    parser.add_argument("--fine_tuning", "-ft", action="store_true")
    parser.add_argument("--log_path", "-l", type=str, required=False, default='training_log.log')
    parser.add_argument("--pretrained_model", "-pm", required=True)
    parser.add_argument("--noisy_data_dir", "-nd", type=str, required=True)
    parser.add_argument("--not_noisy_data_dir", "-nnd", type=str, required=True)
    parser.add_argument("--meta_file", "-m", type=str, required=True)
    parser.add_argument("--only_noise_dir", "-on", type=str, required=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    
    parser.add_argument("--output_path", "-o", type=str, default="./preds_out.csv")
    parser.add_argument("--num_predictions", "-n", type=int, default=300)

    return parser.parse_args()



def validate_args(args : argparse.Namespace) -> None:
    """
    checks if args are valid and returns whether that is true in bool form.
    prints error messages
    """
    if args.fine_tuning and args.pretrained_model == None:
        print("Fatal: fine tuning requested without a path specified to a pretrained model")
        return False
    if args.fine_tuning and not os.path.exists(args.pretrained_model):
        print("Fatal: path to pretrained model does not exist")
        return False

    paths_to_check = (args.noisy_data_dir, args.not_noisy_data_dir, args.meta_file, args.only_noise_dir, args.pretrained_model)
    for path in paths_to_check:
        if not os.path.exists(path):
            raise ValueError(f"path {path} does not exist")
def main():
    args = get_args()
    validate_args(args)
    
    #if verbose:
    print("loading model...")
    model = Model4()
    model.load_state_dict(torch.load(args.pretrained_model))
    data = get_data(args)
    #print(model)
    #metadata = get_metadata(data, 2)
    #print("metadata:", metadata)
    #print_test(data, model)

    #maps = torch.tensor(np.array([get_map(metadata[0]["filename"], data), get_map(metadata[1]["filename"], data)]))
    #print("maps type:", type(maps))
    #print("maps dim:", get_dimensions(maps))
    #print("maps len:", len(maps))
    
    #print("predictions for maps:", model(maps))
    #print("correct radiuses:", metadata[0]["radius"], metadata[1]["radius"])
    #print("correct pitches:", metadata[0]["pitch"], metadata[1]["pitch"])
    #print("correct labels of analysed files:", get_label(metadata[0]), get_label(metadata[1]))
    #for mrcdic in metadata:
    #    if not mrcdic["onlyNoise"]:
    #        print(f"not only noise found!")
    #        break
    #print(data[0])
    benchmark(model, data, args.verbose, args.output_path, args.num_predictions)

main()
