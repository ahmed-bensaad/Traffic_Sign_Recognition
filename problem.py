import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.metrics import precision_score
from rampwf.score_types.classifier_base import ClassifierBaseScoreType


problem_title = 'Traffic Signs Recognition'



Predictions = rw.prediction_types.make_multiclass(
    label_names=list(np.arange(43)))

workflow = rw.workflows.Classifier()

class MacroAveragedPrecision(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='accuracy', precision=4):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        score = precision_score(
            y_true_label_index, y_pred_label_index, average='macro')
        return score



score_types = [
    rw.score_types.Accuracy(name='acc',precision=4),
    rw.score_types.NegativeLogLikelihood(name='negloglik',precision=4),
    rw.score_types.MacroAveragedRecall(name='marecall',precision=4),
    MacroAveragedPrecision(name='maprecision',precision=4),
    rw.score_types.F1Above(name='f1_70', threshold=0.7, precision=4),
    rw.score_types.F1Above(name='f1_80', threshold=0.8, precision=4),

]

def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, typ):
    """
    Read and process data and labels.
    Parameters
    ----------
    path : path to directory that has 'data' subdir
    typ : {'Train', 'Test'}
    Returns
    -------
    X, y data
    """
    test = os.getenv('RAMP_TEST_MODE', 0)


    try:
        data_path = os.path.join(path, 'data',
                                 '{}.csv'.format(typ))

        data = pd.read_csv(data_path)
    except IOError:
        raise IOError("'data/{0}.csv' is not "
                      "found. Run annotations_gen.py to get annotations".format(typ))

    X = data['Filename']
    Y = data['ClassId']

    if test:
        # return src, y
        X1,X2,Y1,Y2 = train_test_split(X,Y, X.shape[0]-100)
        return X2,Y2
    else:
        return X, Y


def get_test_data(path='.'):
    return _read_data(path, 'Test')


def get_train_data(path='.'):
    return _read_data(path, 'Train')

