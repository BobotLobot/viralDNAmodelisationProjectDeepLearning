from trainingWithMeta import Model4
from trainingWithMeta import get_args, are_args_valid
from model import MrcDataset1vMetaDataWithNoiseFile

def get_data():
    args = get_args()
    if not are_args_valid(args):
        raise ValueError()

    # load data from args
    noisyDataDirectoryPath = args.noisy_data_dir
    noNoiseDataDirectoryPath = args.not_noisy_data_dir
    metafile = args.meta_file
    dataset = MrcDataset1vMetaDataWithNoiseFile(metaFile=metafile, noiseDirectory=noisyDataDirectoryPath, noNoiseDirectory=noNoiseDataDirectoryPath, onlyNoise = args.only_noise_dir) # note: copied from other model
    return dataset
    
def main():
    model = Model4()
    #print(model)
    data = get_data()
    print(data[0])
    

main()
