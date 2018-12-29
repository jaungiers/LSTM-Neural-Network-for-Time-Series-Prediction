import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle as pk
import json
from core.data_processor import DataLoader
import os
from random import random

def score(predictions, configs, verbose=False):
    """
    compute a simple binary score on the prediction correctness

    the simple idea behind the score is: this prediction is only on
    the trend, so I will see if the predicted trend in 10 days actually
    correspond to the real outcome direction (up, down) after 10 days

    hyperparams: 
    - predictions are of len PSL = 10
    - index of reference columns (close) RCX = 0

    Args:
        predictions: np.array of predictions, shape(num predictions, pred_len)
        configs: (dict) json load config.json
    """
    print("[SCORE] called")
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )


    def before_prediction(index):
        # returns np.array() of size window_len 
        # sequence of train data prior to prediciton on RCX
        return np.transpose(x[index])[RCX]

    scores_sum = 0
    for index, prediction in enumerate(predictions):

        prev_vals = before_prediction(index)
        delta_y = prev_vals[-1] - predictions[index][0]

        start_val = prev_vals[-1]
        real_val  = y[index+PSL]
        pred_val  = predictions[index][-1] + delta_y

        # this is the vertical distance between last value
        # and first prediction value, used for pad predictions on y

        # the binary score check if pred_val and real_val are in the
        # same of the two parts of plane divided by y = start_val
        binary_score = ((pred_val - start_val) * (real_val - start_val)) >= 0
        if verbose: print("[SCORE] start_val = {}, pred_val = {}, real_val = {}, binary_score = {}".format(start_val, pred_val, real_val, binary_score))
        scores_sum += binary_score

    res_score = scores_sum / len(predictions)
    if verbose: print("[SCORE] res_score = {}".format(res_score))
    return res_score


def visualize(predictions, configs, index):
    """
    called with a (random) index, plots X Y and preds

    hyperparams: 
    - predictions are of len PSL = 10
    - index of reference columns (close) RCX = 0

    Args:
        predictions: np.array of predictions, shape(num predictions, pred_len)
        configs: (dict) json load config.json
    """

    print("[VISUALIZE] called with "+str(index))
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

    # this is the vertical distance between last value
    # and first prediction value, used for pad predictions on y
    delta_y = prev_vals[-1] - predictions[index][0]

    plt.plot(range(len(prev_vals)), prev_vals, 
            label='values before prediction')
    plt.plot(range(len(prev_vals),len(prev_vals)+len(real_vals)),
            real_vals, label='real values of predictions')
    plt.plot(range(len(prev_vals),len(prev_vals)+len(predictions[index])),
            predictions[index] + delta_y, label='model predictions')
    plt.show()

if __name__ == "__main__":
    PSL, RCX = 10, 0
    predictions_name = "prediction.out"
    predictions = pk.load(open(predictions_name,"rb"))
    configs = json.load(open("config.json"))
    curr_score = score(predictions, configs, verbose=False)
    print("[SCORE] score for current prediction = {}".format(curr_score))
    while True:
        print("[SCORE] visualize next? y/n")
        got = input()
        if got == "n": exit("[SCORE] quit")
        random_index = int(random()*len(predictions))
        visualize(predictions, configs, random_index)
