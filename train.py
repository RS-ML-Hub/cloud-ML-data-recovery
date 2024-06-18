import tensorflow as tf
from network.model import GenModel
import numpy as np
import matplotlib.pylab as plt
import utils.prepare_dataset
from losses import lossL1
#TODO Load the data


def main():
    model = GenModel()
    coarseNet = model.coarseNet
    coarseNet.compile(optimizer='adam', loss=lossL1)
    print(coarseNet.summary())


if __name__ == '__main__':
    main()
    