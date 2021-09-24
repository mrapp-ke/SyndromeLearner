#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
import logging as log
from argparse import ArgumentParser
from ast import literal_eval


def log_level(s):
    s = s.lower()
    if s == 'debug':
        return log.DEBUG
    elif s == 'info':
        return log.INFO
    elif s == 'warn' or s == 'warning':
        return log.WARN
    elif s == 'error':
        return log.ERROR
    elif s == 'critical' or s == 'fatal':
        return log.CRITICAL
    elif s == 'notset':
        return log.NOTSET
    raise ValueError('Invalid argument given for parameter \'--log-level\': ' + str(s))


def boolean_string(s):
    s = s.lower()

    if s == 'false':
        return False
    if s == 'true':
        return True
    raise ValueError('Invalid boolean argument given: ' + str(s))


def optional_string(s):
    if s is None or s.lower() == 'none':
        return None
    return s


def optional_dict(s):
    string = optional_string(s)
    return {} if string is None else literal_eval(string)


class ArgumentParserBuilder:
    """
    A builder that allows to configure an `ArgumentParser` that accepts commonly used command-line arguments.
    """

    def __init__(self, description: str, **kwargs):
        parser = ArgumentParser(description=description)
        parser.add_argument('--log-level', type=log_level,
                            default=ArgumentParserBuilder.__get_or_default('log_level', 'info', **kwargs),
                            help='The log level to be used')
        self.parser = parser

    def add_random_state_argument(self, **kwargs) -> 'ArgumentParserBuilder':
        self.parser.add_argument('--random-state', type=int,
                                 default=ArgumentParserBuilder.__get_or_default('random_state', 1, **kwargs),
                                 help='The seed to be used by RNGs')
        return self

    def add_learner_arguments(self, **kwargs) -> 'ArgumentParserBuilder':
        parser = self.parser
        parser.add_argument('--data-dir', type=str, required=True,
                            help='The path of the directory where the data sets are located')
        parser.add_argument('--output-dir', type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('output_dir', None, **kwargs),
                            help='The path of the directory into which results should be written')
        parser.add_argument('--model-dir', type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('model_dir', None, **kwargs),
                            help='The path of the directory where models should be saved')
        parser.add_argument('--dataset', type=str, required=True, help='The name of the data set to be used')
        parser.add_argument('--one-hot-encoding', type=boolean_string,
                            default=ArgumentParserBuilder.__get_or_default('one_hot_encoding', False, **kwargs),
                            help='True, if one-hot-encoding should be used, False otherwise')
        return self

    def add_rule_learner_arguments(self, **kwargs) -> 'ArgumentParserBuilder':
        self.add_learner_arguments(**kwargs)
        self.add_random_state_argument(**kwargs)
        parser = self.parser
        parser.add_argument('--feature-format', type=optional_string, default='auto',
                            help='The format to be used for the feature matrix or \'auto\'')
        parser.add_argument('--num-threads-refinement', type=int,
                            default=ArgumentParserBuilder.__get_or_default('num_threads_refinement', 1, **kwargs),
                            help='The number of threads to be used to search for potential refinements of rules or -1')
        parser.add_argument('--max-rules', type=int,
                            default=ArgumentParserBuilder.__get_or_default('max_rules', 50, **kwargs),
                            help='The maximum number of rules to be induced or -1')
        parser.add_argument('--time-limit', type=int,
                            default=ArgumentParserBuilder.__get_or_default('time_limit', -1, **kwargs),
                            help='The duration in seconds after which the induction of rules should be canceled or -1')
        parser.add_argument('--feature-sub-sampling', type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('feature_sub_sampling', None, **kwargs),
                            help='The name of the strategy to be used for feature sub-sampling or None')
        parser.add_argument('--min-support', type=float,
                            default=ArgumentParserBuilder.__get_or_default('min_support', 0.0001, **kwargs))
        parser.add_argument('--max-conditions', type=int,
                            default=ArgumentParserBuilder.__get_or_default('max_conditions', -1, **kwargs),
                            help='The maximum number of conditions to be included in a rule\'s body or -1')
        parser.add_argument('--print-rules', type=boolean_string,
                            default=ArgumentParserBuilder.__get_or_default('print_rules', True, **kwargs),
                            help='True, if the induced rules should be printed on the console, False otherwise')
        parser.add_argument('--print-options', type=optional_dict,
                            default=ArgumentParserBuilder.__get_or_default('print_options', {}, **kwargs),
                            help='A dictionary that specifies options for printing rules')
        parser.add_argument('--store-rules', type=boolean_string,
                            default=ArgumentParserBuilder.__get_or_default('store_rules', True, **kwargs),
                            help='True, if the induced rules should be stored in text files, False otherwise')
        parser.add_argument('--store-predictions', type=boolean_string,
                            default=ArgumentParserBuilder.__get_or_default('store_predictions', True, **kwargs),
                            help='True, if the predictions should be stored in CSV files, False otherwise')
        return self

    def add_time_series_learner_arguments(self, **kwargs) -> 'ArgumentParserBuilder':
        self.add_rule_learner_arguments(**kwargs)
        parser = self.parser
        parser.add_argument('--temp-dir', type=str, required=True,
                            help='The path of the directory where temporary files are located')
        parser.add_argument('--feature-definition', type=str, required=True,
                            help='The name of the text file that specifies the features to be used')
        parser.add_argument('--from-year', type=int, required=True,
                            help='The first year (inclusive) to be taken from the input data')
        parser.add_argument('--from-week', type=int,
                            default=ArgumentParserBuilder.__get_or_default('from_week', -1, **kwargs),
                            help='The first week (inclusive) of the first year to be taken from the input data')
        parser.add_argument('--to-year', type=int, required=True,
                            help='The last year (inclusive) to be taken from the input data')
        parser.add_argument('--to-week', type=int,
                            default=ArgumentParserBuilder.__get_or_default('to_week', -1, **kwargs),
                            help='The last week (inclusive) of the last year to be taken from the input data')
        parser.add_argument('--count-file-name', type=str,
                            default=ArgumentParserBuilder.__get_or_default('count_file_name', None, **kwargs),
                            help='The name of the file that stores the number of cases for individual weeks')
        return self

    def build(self) -> ArgumentParser:
        return self.parser

    @staticmethod
    def __get_or_default(key: str, default_value, **kwargs):
        return kwargs[key] if key in kwargs else default_value
