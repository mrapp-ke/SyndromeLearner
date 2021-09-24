#!/usr/bin/python


"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
import logging as log
from abc import ABC

from sklearn.base import clone

from rl.common.learners import Learner, NominalAttributeLearner
from rl.testbed.data import MetaData, AttributeType
from rl.testbed.persistence import ModelPersistence
from rl.testbed.printing import ModelPrinter, PredictionPrinter
from rl.testbed.training import CrossValidation, DataSet


class Experiment(CrossValidation, ABC):
    """
    An experiment that trains and evaluates a single multi-label classifier or ranker on a specific data set using cross
    validation or separate training and test sets.
    """

    def __init__(self, base_learner: Learner, data_set: DataSet, num_folds: int = 1, current_fold: int = -1,
                 model_printer: ModelPrinter = None, prediction_printer: PredictionPrinter = None,
                 persistence: ModelPersistence = None):
        """
        :param base_learner:    The classifier or ranker to be trained
        :param model_printer:   The printer that should be used to print textual representations of models
        :param persistence:     The `ModelPersistence` that should be used for loading and saving models
        """
        super().__init__(data_set, num_folds, current_fold)
        self.base_learner = base_learner
        self.model_printer = model_printer
        self.prediction_printer = prediction_printer
        self.persistence = persistence

    def run(self):
        log.info('Starting experiment \"' + self.base_learner.get_name() + '\"...')
        super().run()

    def _train_and_evaluate(self, meta_data: MetaData, train_indices, train_x, train_y, test_indices, test_x, test_y,
                            first_fold: int, current_fold: int, last_fold: int, num_folds: int):
        base_learner = self.base_learner
        current_learner = clone(base_learner)
        learner_name = current_learner.get_name()

        # Set the indices of nominal attributes, if supported...
        if isinstance(current_learner, NominalAttributeLearner):
            current_learner.nominal_attribute_indices = meta_data.get_attribute_indices(AttributeType.NOMINAL)

        # Load model from disc, if possible, otherwise train a new model...
        loaded_learner = self.__load_model(model_name=learner_name, current_fold=current_fold, num_folds=num_folds)

        if isinstance(loaded_learner, Learner):
            current_learner = loaded_learner
        else:
            log.info('Fitting model to %s training examples...', train_x.shape[0])
            current_learner.fit(train_x, train_y)
            log.info('Successfully fit model in %s seconds', current_learner.train_time_)

        # Save model to disk...
        self.__save_model(current_learner, current_fold=current_fold, num_folds=num_folds)

        # Print model, if necessary...
        model_printer = self.model_printer

        if model_printer is not None:
            model_printer.print(learner_name, meta_data, current_learner, current_fold=current_fold,
                                num_folds=num_folds)

        prediction_printer = self.prediction_printer

        if prediction_printer is not None:
            prediction_printer.print(learner_name, current_learner, current_fold=current_fold, num_folds=num_folds)

    def __load_model(self, model_name: str, current_fold: int, num_folds: int):
        """
        Loads the model from disk, if available.

        :param model_name:      The name of the model to be loaded
        :param current_fold:    The current fold starting at 0, or 0 if no cross validation is used
        :param num_folds:       The total number of cross validation folds or 1, if no cross validation is used
        :return: The loaded model
        """
        persistence = self.persistence

        if persistence is not None:
            return persistence.load_model(model_name=model_name, fold=(current_fold if num_folds > 1 else None))

        return None

    def __save_model(self, model: Learner, current_fold: int, num_folds: int):
        """
        Saves a model to disk.

        :param model:           The model to be saved
        :param current_fold:    The current fold starting at 0, or 0 if no cross validation is used
        :param num_folds:       The total number of cross validation folds or 1, if no cross validation is used
        """
        persistence = self.persistence

        if persistence is not None:
            persistence.save_model(model, model_name=model.get_name(), fold=(current_fold if num_folds > 1 else None))
