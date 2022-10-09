from train import Train
import debug as debug
import os

if __name__ == '__main__':
    # datasets = ["librispeech"]
    # datasets = ["nsynth"]
    datasets = ["voxforge"]
    # datasets = ["crema_d"]
    debug.enviromentInfo()
    Train(datasets)
