import tensorflow as tf
from model import GenModel
import numpy as np
import matplotlib.pylab as plt
import utils.prepare_dataset

#TODO Load the data


def main():
    model = GenModel()
    coarseNet = model.coarseNet
    coarseNet.compile(optimizer='adam', loss='mean_squared_error')
    print(coarseNet.summary())


if __name__ == '__main__':
    main()
    