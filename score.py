import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle as pk
import json
from core.data_processor import DataLoader
import os
from random import random

def visualize(predictions_name, configs, index):
    """
    called with a (random) index, plots X Y and preds

    hyperparams: 
    - predictions are of len PSL = 10
    - index of reference columns (close) RCX = 0

    Args:
        predictions: (str) filename
        configs: (dict) json load config.json
    """
    PSL, RCX = 10, 0

    print("[VISUALIZE] called with "+str(index))
    predictions = pk.load(open(predictions_name,"rb"))
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    if not predictions:
        exit("empty predictions")
    print("[VISUALIZE] {} predictions each of size {}".format(len(predictions), len(predictions[0])))
    print("[VISUALIZE] x.shape = {}, y.shape = {}".format(x.shape, y.shape))
    def before_prediction(index):
        # returns np.array() of size window_len 
        # sequence of train data prior to prediciton on RCX
        return np.transpose(x[index])[RCX]

    prev_vals = before_prediction(index)
    real_vals = y[index:index+PSL]
    plt.plot(range(len(prev_vals)), prev_vals, 
            label='values before prediction')
    plt.plot(range(len(prev_vals),len(prev_vals)+len(real_vals)),
            real_vals, label='real values of predictions')
    plt.plot(range(len(prev_vals),len(prev_vals)+len(predictions[index])),
            predictions[index], label='model predictions')
    plt.show()

if __name__ == "__main__":
    predictions_name = "prediction.out"
    configs = json.load(open("config.json"))
    while True:
        print("[SCORE] visualize next? y/n")
        got = input()
        if got == "n": exit("[SCORE] quit")
        random_index = int(random()*200)
        visualize(predictions_name, configs, random_index)
