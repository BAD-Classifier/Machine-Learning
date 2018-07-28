import numpy, scipy, matplotlib.pyplot as plt, librosa, sklearn
import urllib.request
import librosa.display
import os

def convert_to_image(birdSoundPath, birdName):
    x, fs = librosa.load(birdSoundPath,sr=None,mono=True)
    mfccs = librosa.feature.mfcc(x, sr=fs, n_fft=1024, hop_length=512, n_mfcc=13)
    librosa.display.specshow(mfccs, sr=fs, x_axis='time')
    mfccs = sklearn.preprocessing.scale(mfccs, axis=1)  
    mfccs.mean(axis=1)
    mfccs.var(axis=1)
    librosa.display.specshow(mfccs, sr=fs*2, cmap='coolwarm')
    picName = birdName[:-4] + '.png'
    save_image(picName)

def save_image(picName):
    path = os.getcwd() + '/BirdMFCCS/Short/'
    if not os.path.exists(path):
        os.makedirs(path)
    fileName = path + picName
    plt.savefig(fileName, bbox_inches='tight', pad_inches=0)


def main():
    plt.rcParams['figure.figsize'] = (14,4)

    totalFiles = 672
    count = 1

    path = os.getcwd() + '/Sounds/'
    fileNames = os.listdir(path)
    for fileName in fileNames:
        birdSound = path + fileName
        print(str(count) + '/' + str(totalFiles) + ' ' + fileName + ' conversion has started')
        convert_to_image(birdSound, fileName)
        print(str(count) + '/' + str(totalFiles) + ' ' + fileName + ' MFCC has been generated')
        count = count + 1

if __name__ == "__main__":
    main()