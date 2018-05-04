from enum import Enum


class Datafile(Enum):
    SimulatedLm10 = 'lm_10.csv'

    # Predict forest fire area
    # http://archive.ics.uci.edu/ml/datasets/Forest+Fires
    ForestFire = 'forestfires.csv'
