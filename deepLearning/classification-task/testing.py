import torch
from model import ModelClassificationMultival as Model
from training import get_args, are_args_valid
from model import MrcDataset1vMetaDataWithNoiseFile, DataLoader

def get_data():
    args = get_args()
    if not are_args_valid(args):
        raise ValueError()

    # load data from args
    noisyDataDirectoryPath = args.noisy_data_dir
    noNoiseDataDirectoryPath = args.not_noisy_data_dir
    metafile = args.meta_file
    
    dataset = MrcDataset1vMetaDataWithNoiseFile(metaFile=metafile,
    noiseDirectory=noisyDataDirectoryPath,
    noNoiseDirectory=noNoiseDataDirectoryPath,
    onlyNoise = args.only_noise_dir)
    
    return dataset
    
def main():
    pretrained_model_path = "./out/best_model.pth"
    model = Model()
    model.load_state_dict(torch.load(pretrained_model_path))
    #print(model)
    
    data = get_data()
    #print(data[0])

    trainDataloader = DataLoader(data, batch_size=3, shuffle=True)
    for i, data in enumerate(trainDataloader,0):
        inputs, labels = data
        inputs = inputs.float()
        labels = labels
        outputs = model(inputs)
        print("outputs:", outputs)
        return
    
main()
