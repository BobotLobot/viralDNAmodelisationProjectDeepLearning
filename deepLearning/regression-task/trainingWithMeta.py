from model import *
import time
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from argparse import ArgumentParser
from model import MrcDataset1vMetaDataWithNoiseFile

def save_losses_to_csv(train_losses, valid_losses, file_path='losses.csv')->None:
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
            for epoch, (train_loss, valid_loss) in enumerate(zip(train_losses, valid_losses)):
                writer.writerow([epoch + 1, train_loss, valid_loss])

def plot_training_losses(train_losses, valid_losses, save_path='training3D.png') -> None:
    """
    Plot training and validation losses and save to file.
    
    Args:
        train_losses (list): List of training losses
        valid_losses (list): List of validation losses
        save_path (str): Path to save the plot image
    """
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def are_args_valid(args) -> bool:
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

    paths_to_check = (args.noisy_data_dir, args.not_noisy_data_dir, args.meta_file)
    for path in paths_to_check:
        if not os.path.exists(path):
            print(f"Fatal: path {path} does not exist")
            return False

    if args.max_epochs <= 0:
        raise ValueError("max epochs must be a positive integer")
    if args.learning_rate < 0 or args.learning_rate > 1:
        raise ValueError("learning rate must be floating-point number between 0 and 1")
    if args.noise_chance < 0 or args.noise_chance > 1:
        raise ValueError("noise chance must be floating-point number between 0 and 1")
    if args.dropout_factor < 0 or args.dropout_factor > 1:
        raise ValueError("dropout factor must be floating-point number between 0 and 1")
    if args.weight_decay < 0 or args.weight_decay > 1:
        raise ValueError("weight decay must be floating-point number between 0 and 1")
    
    return True

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--fine_tuning", "-ft", action="store_true")
    parser.add_argument("--pretrained_model", "-pm", type=str)
    parser.add_argument("--noisy_data_dir", "-nd", type=str, required=True)
    parser.add_argument("--not_noisy_data_dir", "-nnd", type=str, required=True)
    parser.add_argument("--meta_file", "-m", type=str, required=True)
    parser.add_argument("--loss_output", "-lo", type=str, default="training_loss.png")
    parser.add_argument("--model_output", "-mo", type=str, default='best_model.pth')
    parser.add_argument("--patience", "-p", type=int, default=20)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--max_epochs", "-me", type=int, default=1000)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.05)
    parser.add_argument("--noise_chance", "-nc", type=float, default=0.95)
    parser.add_argument("--momentum", "-mom", type=float, default=0.9)
    parser.add_argument("--dropout_factor", "-df", type=float, default=1.0)
    parser.add_argument("--weight_decay", "-wd", type=float, default=0.01)
    
    return parser.parse_args()

def main() -> None:
    args = get_args()
    if not are_args_valid(args):
        return

    #set randomness
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    deviceName= torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Current cuda device: {deviceName}")

    #--------------------------------------Fine-tuning
    fineTunning = args.fine_tuning
    pretrained_model_path = args.pretrained_model

    #--------------------------------------Data directory
    noisyDataDirectoryPath = args.noisy_data_dir
    noNoiseDataDirectoryPath = args.not_noisy_data_dir
    metafile = args.meta_file
    #-------------------------------------- Data loading
    #Noisy data and non noisy data are loaded separately
    #After that  a part of the noisy are split in order to make test dataset
    #The rest of the noisy Dataset and the noNoiseData #are concatenated

    torch.cuda.empty_cache() #maybe not necessary just in case
    #loading class
    dataset = MrcDataset1vMetaDataWithNoiseFile(metaFile=metafile,
    noiseDirectory=noisyDataDirectoryPath,
    noNoiseDirectory=noNoiseDataDirectoryPath,
    verbose = args.verbose,
    odds_noisy = args.noise_chance)
    #splitting
    trainDataset, testDataset = random_split(dataset , [0.8, 0.2])
    
    print(f"number of data points in training: {len(trainDataset)}")
    print(f"number of data points in testing: {len(testDataset)}")
    #data loader
    batchSize= 200 # => must be max GPU node memory
    #worker are max CPU core of the GPU node
    trainDataloader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testDataloader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)



    #--------------------------------------model initiaitsation & learning parameters

    model= Model4(dropout_factor = args.dropout_factor).to(device)
    loss_fn = torch.nn.MSELoss()
    
    optimizer = torch.optim.SGD(
    model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum = args.momentum
    )
#    optimizer = torch.optim.AdamW(
#        model.parameters(),
#        lr=args.learning_rate,
#        betas=(0.9, 0.9999),
#        weight_decay=0.01,
#        eps=1e-7,
#        momentum = args.momentum
       #     amsgrad=True
#    )

   # scheduler = torch.optim.lr_scheduler.CyclicLR(
   #     optimizer,
   #     base_lr=0.0001,
   #     max_lr=0.01,
   #     step_size_up=2000,
   #     cycle_momentum=False
   # )

    scaler = GradScaler()
    # Load the pre-trained model
    if fineTunning:
        try:
            model.load_state_dict(torch.load(pretrained_model_path))
            print(f"Loaded pre-trained model from {pretrained_model_path}")
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")

    running_loss = 0.
    last_loss = 0.

    for inputs, labels in trainDataloader:
        print(f"Input shape: {inputs.shape}")
        print(f"Label shape: {labels.shape}")
        break

    #learning variable
    running_losses = []
    losses = []
    best_vloss = 1_000_000.
    best_tloss = 1_000_000.
    trainLosess=[]
    validLosess=[]
    waited=0

    #--------------------------------------Training
    print("Start training")
    timeTraining = time.time()
    for epoch in range(args.max_epochs):
        timeEpoch = time.time()
        print(f"Epoch {epoch} out of {args.max_epochs}")
        model.train(True)
        print(f"number of batch: {len(trainDataloader)}")
        for i, data in enumerate(trainDataloader,0):
            inputs, labels = data
            #print("inputs:", inputs[0])
            print("labels:", labels[0])
            #return
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            print("outputs:", outputs[0])
            loss = loss_fn(outputs, labels) # MSE between outputs and labels
            single_loss = loss_fn(outputs[0], labels[0])
            print("loss:", single_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #gradient clipping
            optimizer.step()
            running_loss += loss.item()
            if i == len(trainDataloader) - 1:
                print("------------------------------------------------")
                last_loss = running_loss/len(trainDataloader)
                trainLosess.append(last_loss)
                print(f"Epoch {epoch}, loss: {last_loss:.4f}")
                running_loss = 0.0
        #print(f"LR: {optimizer.param_groups[0]['lr']}")
        print("evaluating")
        model.eval()
        vrunning_loss = 0.
        predictions = []
        actuals = []
        with torch.no_grad(): # disable gradient calculation for testing
            for i, vdata in enumerate(testDataloader, 0):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs = model(vinputs)
                predictions.extend(voutputs.cpu().numpy())
                actuals.extend(vlabels.cpu().numpy())
                vloss = loss_fn(voutputs, vlabels)
                vrunning_loss += vloss.item()
        avg_vloss = vrunning_loss / len(testDataloader)
        validLosess.append(avg_vloss)
        #scheduler.step()
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_tloss = last_loss
            torch.save(model.state_dict(), args.model_output)
            print("saving model state dict to", args.model_output)
        else:
            waited+=1
            print(f"{waited} epochs have passed since validation loss last dropped. patience: {args.patience}")
            if waited > args.patience:
                print(f"LOSS train {last_loss} valid {avg_vloss}")
                print(f"Time for epoch {epoch}: {time.time()-timeEpoch}")
                print("Early stopping")
                break
        
        print(f"LOSS train {last_loss} valid {avg_vloss}")
        print(f"Time for epoch {epoch}: {time.time()-timeEpoch}")

        plot_training_losses(trainLosess, validLosess, save_path=args.loss_output)

        print(f"Time for training over {epoch} epoch: {time.time()-timeTraining}")
    save_losses_to_csv(trainLosess, validLosess)
    print("====================================================================")
    print("end of the training")
    print(f"LOSS train {best_tloss} valid {best_vloss}")

if __name__ == "__main__":
    main()

