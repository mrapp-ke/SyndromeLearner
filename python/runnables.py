#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
import logging as log
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser

from rl.testbed.experiments import Experiment
from rl.testbed.persistence import ModelPersistence
from rl.testbed.printing import RulePrinter, ModelPrinterLogOutput, ModelPrinterTxtOutput, PredictionPrinter, \
    PredictionPrinterCsvOutput
from rl.testbed.training import DataSet

LOG_FORMAT = '%(levelname)s %(message)s'


class Runnable(ABC):
    """
    A base class for all programs that can be configured via command line arguments.
    """

    def run(self, parser: ArgumentParser):
        args = parser.parse_args()

        # Configure the logger...
        log_level = args.log_level
        root = log.getLogger()
        root.setLevel(log_level)
        out_handler = log.StreamHandler(sys.stdout)
        out_handler.setLevel(log_level)
        out_handler.setFormatter(log.Formatter(LOG_FORMAT))
        root.addHandler(out_handler)

        log.info('Configuration: %s', args)
        self._run(args)

    @abstractmethod
    def _run(self, args):
        """
        Must be implemented by subclasses in order to run the program.

        :param args: The command line arguments
        """
        pass


class RuleLearnerRunnable(Runnable, ABC):
    """
    A base class for all programs that perform an experiment that involves training and evaluation of a rule learner.
    """

    def _run(self, args):
        model_printer_outputs = []
        prediction_printer_outputs = []
        output_dir = args.output_dir

        if args.print_rules:
            model_printer_outputs.append(ModelPrinterLogOutput())

        if output_dir is not None:
            if args.store_rules:
                model_printer_outputs.append(ModelPrinterTxtOutput(output_dir=output_dir))

            if args.store_predictions:
                prediction_printer_outputs.append(PredictionPrinterCsvOutput(output_dir=output_dir, clear_dir=False))

        model_dir = args.model_dir
        persistence = None if model_dir is None else ModelPersistence(model_dir)
        learner = self._create_learner(args)
        model_printer = RulePrinter(args.print_options, model_printer_outputs) if len(
            model_printer_outputs) > 0 else None
        prediction_printer = PredictionPrinter(prediction_printer_outputs) if len(
            prediction_printer_outputs) > 0 else None
        if args.data_dir is None:
            raise ValueError('Mandatory parameter \'--data-dir\' has not been specified')
        if args.dataset is None:
            raise ValueError('Mandatory parameter \'--dataset\' has not been specified')
        data_dir, dataset = self._preprocess(args)
        data_set = DataSet(data_dir=data_dir, data_set_name=dataset, use_one_hot_encoding=args.one_hot_encoding)
        experiment = Experiment(learner, data_set=data_set, model_printer=model_printer,
                                prediction_printer=prediction_printer, persistence=persistence)
        experiment.random_state = args.random_state
        experiment.run()

    @abstractmethod
    def _create_learner(self, args):
        """
        Must be implemented by subclasses in order to create the learner.

        :param args:    The command line arguments
        :return:        The learner that has been created
        """
        pass

    @abstractmethod
    def _preprocess(self, args) ->(str, str):
        """
        Must be implemented by subclasses in order to preprocess the data.

        :param args:    The command line arguments
        :return:        The path to the directory where the preprocessed data is located, as well as the file of the
                        dataset (without suffix)
        """
        pass
