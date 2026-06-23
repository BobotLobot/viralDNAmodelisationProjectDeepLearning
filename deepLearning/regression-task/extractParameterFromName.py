import os
import re
import mrcfile
import numpy as np

def get_radius_value(filename):
    match = re.search(r'radius(\d+.\d+)', filename)
    if match:
        return float(match.group(1))
    return None

def get_radius_valueint(filename):
    match = re.search(r'radius(\d+)', filename)
    if match:
        return float(match.group(1))
    return None

def get_pitch_value(filename):
    match = re.search(r'pitch(\d)', filename)
    match_decimal = re.search(r'pitch(\d+\.\d+)', filename)
    if match_decimal:
        return float(match_decimal.group(1))

def get_phi_value(filename):
    match = re.search(r'phi(\d+)', filename)
    if match:
        return float(match.group(1))
    return None

def getMrcFeatures(directory="/cephyr/users/robincoh/Alvis/viralTracing/dataset/spiralMidlePitchWithNoises")->dict:
    fileFeatures = {}
    listFeatures = []
    for filename in os.listdir(directory):
        if filename.endswith(".mrc"):
            #get radius
            radius = get_radius_value(filename)
            if radius is None:
                radius=get_radius_valueint(filename)
            pitch= get_pitch_value(filename)
            phi = 3
            if any(v is None for v in [filename,radius, pitch]):
                continue
            listFeatures.append([os.path.join(directory, filename), radius, pitch])
    for i in range(len(listFeatures)):
        fileFeatures[i] = {"filename": listFeatures[i][0], "radius": listFeatures[i][1], "pitch": listFeatures[i][2]}
    return fileFeatures

def getMrcFeaturesPhi(directory="/cephyr/users/robincoh/Alvis/viralTracing/dataset")->dict:
    fileFeatures = {}
    listFeatures = []
    for filename in os.listdir(directory):
        if filename.endswith(".mrc"):

            #get radius
            radius = get_radius_value(filename)
            if radius is None:
                radius=get_radius_valueint(filename)
            pitch= get_pitch_value(filename)
            phi = get_phi_value(filename)
            if any(v is None for v in [filename,radius, pitch,phi]):
                continue
            listFeatures.append([os.path.join(directory, filename), radius, pitch,phi])
    for i in range(len(listFeatures)):
        fileFeatures[i] = {"filename": listFeatures[i][0], "radius": listFeatures[i][1], "pitch": listFeatures[i][2],
                           "phi":listFeatures[i][3]}
    return fileFeatures

def get_corruptedFile(mrcDir):
    corruptedFile=[]
    for filename in os.listdir(mrcDir):
        if filename.endswith(".mrc"):
            try:
                with mrcfile.open(os.path.join(mrcDir, filename), mode='r+') as mrc:
                    mrcData = mrc.data
            except:
                print(f"Error loading file {filename}")
                corruptedFile.append(filename)
    return corruptedFile

if __name__ == "__main__":
    noisyDataDirectoryPath="/cephyr/users/robincoh/Alvis/viralTracing/dataset/trainingDataset"
    get_corruptedFile(noisyDataDirectoryPath)
