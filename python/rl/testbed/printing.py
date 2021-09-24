#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
import logging as log
from abc import ABC, abstractmethod
from typing import Dict, List

from rl.common.cython.model import RuleModelFormatter
from rl.common.cython.rule_induction import Predictions
from rl.common.learners import Learner
from rl.testbed.data import MetaData
from rl.testbed.io import clear_directory, open_writable_txt_file, open_writable_csv_file, create_csv_dict_writer


class PredictionPrinterOutput(ABC):

    @abstractmethod
    def write_predictions(self, experiment_name: str, predictions: Predictions, num_folds: int, fold: int = None):
        pass


class PredictionPrinter(ABC):

    def __init__(self, outputs: List[PredictionPrinterOutput]):
        self.outputs = outputs

    def print(self, experiment_name: str, learner: Learner, current_fold: int, num_folds: int):
        if hasattr(learner, 'predictions_'):
            predictions = learner.predictions_

            if isinstance(predictions, Predictions):
                for output in self.outputs:
                    output.write_predictions(experiment_name, predictions, num_folds,
                                             current_fold if num_folds > 1 else None)


class PredictionPrinterCsvOutput(PredictionPrinterOutput):

    def __init__(self, output_dir: str, clear_dir: bool = True):
        self.output_dir = output_dir
        self.clear_dir = clear_dir

    def __clear_dir_if_necessary(self):
        """
        Clears the output directory, if necessary.
        """
        if self.clear_dir:
            clear_directory(self.output_dir)
            self.clear_dir = False

    def write_predictions(self, experiment_name: str, predictions: Predictions, num_folds: int, fold: int = None):
        self.__clear_dir_if_necessary()
        num_rows = len(predictions.ground_truth)
        num_iterations = len(predictions.predictions)
        header = ['Ground truth']

        for i in range(num_iterations):
            header.append('Predictions Iteration ' + str(i + 1))

        with open_writable_csv_file(self.output_dir, 'predictions_' + experiment_name, fold, append=True) as csv_file:
            csv_writer = create_csv_dict_writer(csv_file, header)

            for i in range(num_rows):
                row = {header[0]: predictions.ground_truth[i]}

                for j in range(num_iterations):
                    row[header[1 + j]] = predictions.predictions[j][i]

                csv_writer.writerow(row)


class ModelPrinterOutput(ABC):
    """
    An abstract base class for all outputs, textual representations of models may be written to.
    """

    @abstractmethod
    def write_model(self, experiment_name: str, model: str, total_folds: int, fold: int = None):
        """
        Write a textual representation of a model to the output.

        :param experiment_name:     The name of the experiment
        :param model:               The textual representation of the model
        :param total_folds:         The total number of folds
        :param fold:                The fold for which the results should be written or None, if no cross validation is
                                    used or if the overall results, averaged over all folds, should be written
        """
        pass


class ModelPrinter(ABC):
    """
    An abstract base class for all classes that allow to print a textual representation of a `MLLearner`'s model.
    """

    def __init__(self, options: Dict, outputs: List[ModelPrinterOutput]):
        """
        :param options: A dictionary that contains the options to be used for printing models
        :param outputs: The outputs, the textual representations of models should be written to
        """
        self.options = options
        self.outputs = outputs

    def print(self, experiment_name: str, meta_data: MetaData, learner: Learner, current_fold: int, num_folds: int):
        """
        Prints a textual representation of a `MLLearner`'s model.

        :param experiment_name: The name of the experiment
        :param meta_data:       The meta data of the training data set
        :param learner:         The learner
        :param current_fold:    The current cross validation fold starting at 0, or 0 if no cross validation is used
        :param num_folds:       The total number of cross validation folds or 1, if no cross validation is used
        """
        model = learner.model_
        text = self._format_model(meta_data, model)

        for output in self.outputs:
            output.write_model(experiment_name, text, num_folds, current_fold if num_folds > 1 else None)

    @abstractmethod
    def _format_model(self, meta_data: MetaData, model) -> str:
        """
        Must be implemented by subclasses in order to create a textual representation of a model.

        :param meta_data:   The meta data of the training data set
        :param model:       The model
        :return:            The textual representation of the given model
        """
        pass


class ModelPrinterLogOutput(ModelPrinterOutput):
    """
    Outputs the textual representation of a model using the logger.
    """

    def write_model(self, experiment_name: str, model: str, total_folds: int, fold: int = None):
        msg = 'Model for experiment \"' + experiment_name + '\"' + (
            ' (Fold ' + str(fold + 1) + ')' if fold is not None else '') + ':\n\n%s'
        log.info(msg, model)


class ModelPrinterTxtOutput(ModelPrinterOutput):
    """
    Writes the textual representation of a model to a text file.
    """

    def __init__(self, output_dir: str, clear_dir: bool = True):
        self.output_dir = output_dir
        self.clear_dir = clear_dir

    def write_model(self, experiment_name: str, model: str, total_folds: int, fold: int = None):
        self.__clear_dir_if_necessary()
        with open_writable_txt_file(self.output_dir, 'rules_' + experiment_name, fold, append=False) as text_file:
            text_file.write(model)

    def __clear_dir_if_necessary(self):
        """
        Clears the output directory, if necessary.
        """
        if self.clear_dir:
            clear_directory(self.output_dir)
            self.clear_dir = False


class RulePrinter(ModelPrinter):
    """
    Allows to print a textual representation of a `MLRuleLearner`'s rule-based model.
    """

    def __init__(self, options: Dict, outputs: List[ModelPrinterOutput]):
        super().__init__(options, outputs)

    def _format_model(self, meta_data: MetaData, model) -> str:
        formatter = RuleModelFormatter(attributes=meta_data.attributes, labels=meta_data.labels, **self.options)
        formatter.format(model)
        return formatter.get_text()
