import numpy as np


class MovingAverage:

    def __init__(self, ws):
        self.data = []

    def __call__(self, value):
        try:
            self.data.pop(0)
        except IndexError:
            pass

        self.data.append(value)
        return np.mean(self.data)

    def mean_first(self, value):
        if self.data:
            mean = np.mean(self.data)
            self.data.pop(0)
            self.data.append(value)
            return mean

        else:
            self.data.append(value)
            return value
