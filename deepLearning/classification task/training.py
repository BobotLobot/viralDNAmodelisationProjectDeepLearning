from model import *
import time
import matplotlib.pyplot as plt
import logging
from torch.cuda.amp import autocast, GradScaler
import csv
from argparse import ArgumentParser
import os

def save_losses_to_csv(train_losses, valid_losses, file_path='losses.csv')->None:
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
        for epoch, (train_loss, valid_loss) in enumerate(zip(train_losses, valid_losses)):
            writer.writerow([epoch + 1, train_loss, valid_loss])

def save_accuracy(valid_accuracy_vals, train_accuracy_vals, file_path='accuracy.csv') -> None:
    with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'valid accuracy', 'train accuracy'])
            for epoch, (valid_acc, train_acc) in enumerate(zip(valid_accuracy_vals, train_accuracy_vals)):
                writer.writerow([epoch + 1, valid_acc, train_acc])

def accuracy_fn(label, prediction):
    probabilities = torch.exp(prediction) / torch.sum(torch.exp(prediction))
    return acc

def calculate_accuracy(outputs, labels):
    """
    takes model outputs and correct labels
    returns tuple (number of correct predictions, total number of predictions)
    """
    _, preds = torch.max(outputs, dim=1) # prediction is made here, notably without softmax.
    correct = torch.sum(preds == labels).item()
    total = labels.size(0)
    return correct, total

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

    paths_to_check = (args.noisy_data_dir, args.not_noisy_data_dir, args.meta_file, args.only_noise_dir)
    for path in paths_to_check:
        if not os.path.exists(path):
            print(f"Fatal: path {path} does not exist")
            return False
    return True

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--fine_tuning", "-ft", action="store_true")
    parser.add_argument("--log_path", "-l", type=str, required=False, default='training_log.log')
    parser.add_argument("--pretrained_model", "-pm", type=str)
    parser.add_argument("--noisy_data_dir", "-nd", type=str, required=True)
    parser.add_argument("--not_noisy_data_dir", "-nnd", type=str, required=True)
    parser.add_argument("--meta_file", "-m", type=str, required=True)
    parser.add_argument("--only_noise_dir", "-on", type=str, required=True)
    parser.add_argument("--accuracy_output", "-ao", type=str, default="training_accuracy.png")
    parser.add_argument("--loss_output", "-lo", type=str, default="training_loss.png")
    parser.add_argument("--model_output", "-mo", type=str, default='best_model.pth')

    return parser.parse_args()

def main():
    #parse and check arguments
    args = get_args()
    if not are_args_valid(args):
        return
    
    # Configure logging
    logging.basicConfig(filename=args.log_path, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Replace all print statements with logging.info
    # print = logging.info
    
    #set randomness
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    deviceName= torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Current cuda device: {deviceName}")


    
    #--------------------------------------Fine Tunnig
    fineTunning = args.fine_tuning
    pretrained_model_path = args.pretrained_model # old path: '/cephyr/users/robincoh/Alvis/viralTracing/script/train3/best_model.pth'
    
    #--------------------------------------Data directory
    # dataDirectoryPath="/cephyr/users/robincoh/Alvis/viralTracing/dataset/spiralMidlePitchWithNoises/"
    # dataDirectoryPath="/cephyr/users/robincoh/Alvis/viralTracing/dataset/spiralNoNoisePitchPhiRadius"
    noisyDataDirectoryPath = args.noisy_data_dir #"/mimer/NOBACKUP/groups/naiss2025-23-223/compileNoise"
    noNoiseDataDirectoryPath = args.not_noisy_data_dir #"/mimer/NOBACKUP/groups/naiss2025-23-223/compileNoNoise"
    metafile = args.meta_file # "/mimer/NOBACKUP/groups/naiss2025-23-223/compile.csv"
    onlyNoiseDir = args.only_noise_dir # "/mimer/NOBACKUP/groups/naiss2025-23-223/onlyNoiseNoStruct"
    #-------------------------------------- Data loading
    #Noisy data and non noisy data are loaded separately
    #After that  a part of the noisy are split in order to make test dataset
    #The rest of the noisy Dataset and the noNoiseData #are concatenated
    
    torch.cuda.empty_cache() #maybe not necessary just in case
    #loading class
    dataset = MrcDataset1vMetaDataWithNoiseFile(metaFile=metafile,noiseDirectory=noisyDataDirectoryPath,noNoiseDirectory=noNoiseDataDirectoryPath, onlyNoise=onlyNoiseDir)
    # datasetNoNoise = MrcDataset1vMetaData(getMrcFeatures(noNoiseDataDirectoryPath))

    #splitting
    trainDataset, testDataset = random_split(dataset , [0.8, 0.2])

    #no augmentation/denoising

    print(f"number of data points in training: {len(trainDataset)}")
    print(f"number of data points in testing: {len(testDataset)}")
    #data loader
    batchSize= 200 # => must be max GfPU node memory 
    #worker are max CPU core of the GPU node
    trainDataloader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True) #originally for both: pin_memory=True, num_workers=16
    testDataloader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)



    #--------------------------------------model initiaitsation & learning parameters
    model= ModelClassificationMultival().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.005,
    betas=(0.9, 0.9999),
    weight_decay=0.01,
    eps=1e-7,
    amsgrad=True
    )

    scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=0.0001,
    max_lr=0.01,
    step_size_up=2000,
    cycle_momentum=False
    )

    # Load the pre-trained model
    if fineTunning:
        try:
                model.load_state_dict(torch.load(pretrained_model_path))
                print(f"Loaded pre-trained model from {pretrained_model_path}")
        except Exception as e:
                print(f"Error loading pre-trained model: {e}")
                return

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
    vrunning_accuracy=[]
    running_train_accuracy = []
    patience = 50
    waited=0

    #--------------------------------------Training 
    print("Start training")
    timeTraining = time.time()
    for epoch in range(1000):
        running_loss = 0.
        running_correct = 0
        running_total = 0
        timeEpoch = time.time()
        print(f"Epoch {epoch}")
        model.train(True)
        print(f"number of batch: {len(trainDataloader)}")
        for i, data in enumerate(trainDataloader,0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.squeeze()
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            loss.backward()

	        
	        
            optimizer.step()

            running_loss += loss.item()

            correct, total = calculate_accuracy(outputs, labels)
            running_correct += correct
            running_total += total
	        
	        
            #print(f"Batch: Epoch {epoch}, iter {i}, loss: {loss.item():.4f}")
            if i == len(trainDataloader) - 1:
                print("------------------------------------------------")
                last_loss = running_loss/len(trainDataloader)
                trainLosess.append(last_loss)
                epoch_acc = running_correct / running_total
                running_train_accuracy.append(epoch_acc)
                print(f"Epoch {epoch+1:03d} | "
                f"Train Loss: {last_loss:.4f} | "
                f"Acc: {epoch_acc:.2%} | "
                f"LR: {optimizer.param_groups[0]['lr']}")
                
        print("evaluating model")
        model.eval()
        vrunning_loss = 0.
        valid_correct = 0
        valid_total = 0
        predictions = []
        actuals = []
        with torch.no_grad(): # disable gradient calculation for testing
            for i, vdata in enumerate(testDataloader, 0):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                vlabels = vlabels.squeeze()
                voutputs = model(vinputs)
                predictions.extend(voutputs.cpu())
                actuals.extend(vlabels.cpu())
                vloss = loss_fn(voutputs, vlabels)
                vrunning_loss += vloss.item()
                correct, total = calculate_accuracy(voutputs, vlabels)
                valid_correct += correct
                valid_total += total
        valid_acc = valid_correct / valid_total
        vrunning_accuracy.append(valid_acc)
        avg_vloss = vrunning_loss / len(testDataloader)
        validLosess.append(avg_vloss)
        scheduler.step()
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_tloss = last_loss # actualy not the better overall but associated to the better validation loss
            torch.save(model.state_dict(), args.model_output)
	            
            waited=0
        else:
            waited+=1
            if waited > patience:

                print(f"LOSS train {last_loss} valid {avg_vloss} accuracy {valid_acc}")
                print(f"Time for epoch {epoch}: {time.time()-timeEpoch}")
                print("Early stopping")
                break
        print(f"Validation  | Loss: {avg_vloss:.4f} | Acc: {valid_acc:.2%}")
        print(f"Time for epoch {epoch}: {time.time()-timeEpoch}")

        plt.plot(trainLosess, label='Training Loss')
        plt.plot(validLosess, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('training.png')
        plt.close()

        plt.plot(vrunning_accuracy, label='Valid accuracy')
        plt.plot(running_train_accuracy, label='Train accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Classification accuracy')
        plt.legend()
        plt.savefig('accuracy.png')
        plt.close()


        # save_losses_to_csv(trainLosess, validLosess)
        print(f"Time for training over {epoch} epoch: {time.time()-timeTraining}")
    save_losses_to_csv(trainLosess, validLosess)
    save_accuracy(vrunning_accuracy,running_train_accuracy)
    print("====================================================================")
    print("end of the training")
    print(f"LOSS train {best_tloss} valid {best_vloss}")

if __name__ == "__main__":
    main()
