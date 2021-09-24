#!/usr/bin/python

import logging as log
import os
import xml.etree.ElementTree as XmlTree
from xml.dom import minidom

import arff
import numpy as np
import pandas as pd

from args import ArgumentParserBuilder
from rl.tsa.syndrome_learner import SyndromeLearner
from runnables import RuleLearnerRunnable

ENCODING = 'utf-8'

COLUMN_YEAR = 'year'

COLUMN_WEEK = 'week'

COLUMN_CASES = 'cases'


def __load_counts(input_dir: str, input_file_name: str, from_year: int, from_week: int, to_year: int, to_week: int):
    input_file = os.path.join(input_dir, input_file_name + '.csv')
    log.info(
        'Loading time series data (for the timespan ' + str(from_year) + ('' if from_week < 0 else '-' + str(from_week))
        + ' to ' + str(to_year) + ('' if to_week < 0 else '-' + str(to_week)) + ') from file \'' + str(input_file)
        + '\'...')
    df = pd.read_csv(input_file, encoding=ENCODING)

    if COLUMN_YEAR not in df.columns:
        raise ValueError('Column \'' + COLUMN_YEAR + '\' missing from file \'' + str(input_file) + '\'')
    if COLUMN_WEEK not in df.columns:
        raise ValueError('Column \'' + COLUMN_WEEK + '\' missing from file \'' + str(input_file) + '\'')
    if COLUMN_CASES not in df.columns:
        raise ValueError('Column \'' + COLUMN_CASES + '\' missing from file \'' + str(input_file) + '\'')

    result = {}

    for i, row in df.iterrows():
        year = int(row[COLUMN_YEAR])

        if from_year <= year <= to_year:
            week = int(row[COLUMN_WEEK])

            if (from_week < 0 or from_year != year or week >= from_week) and (
                    to_week < 0 or to_year != year or week <= to_week):
                cases = int(row[COLUMN_CASES])
                result[str(year) + '-' + str(week)] = cases

    log.info('Time series data was loaded successfully!')
    return result


def __load_feature_names(input_dir: str, input_file_name: str):
    input_file_name = input_file_name + '.txt'
    input_path = os.path.join(input_dir, input_file_name)
    log.info('Loading features names from file \'' + str(input_path) + '\'...')

    with open(input_path, 'r') as input_file:
        feature_names = input_file.readlines()
        feature_names = [name.strip() for name in feature_names]
        log.info('Feature names ' + str(feature_names) + ' were loaded successfully!')
        return feature_names


def __load_instances(input_dir: str, input_file_name: str, from_year: int, from_week: int, to_year: int, to_week: int,
                     counts, feature_names):
    input_file = os.path.join(input_dir, input_file_name + '.csv')
    log.info('Loading instances from file \'' + str(input_file) + '\'...')
    df = pd.read_csv(input_file, encoding=ENCODING)
    feature_names.append(COLUMN_WEEK)

    for name in feature_names:
        if name not in df.columns:
            raise ValueError('Column \'' + name + '\' missing from file \'' + str(input_file) + '\'')

    # Remove unused columns...
    df = df[feature_names]

    # Filter by year (and week)...
    df[COLUMN_YEAR] = df[COLUMN_WEEK].apply(lambda x: int(x.split('-')[0]))
    df['tmp'] = df[COLUMN_WEEK].apply(lambda x: int(x.split('-')[1]))
    to_year_incl = to_year + 1
    df = df[(df[COLUMN_YEAR].isin(range(from_year, to_year_incl))) &
            (from_week < 0 or df[COLUMN_YEAR].ne(from_year) or df[COLUMN_WEEK].ge(from_week)) &
            (to_week < 0 or df[COLUMN_YEAR].ne(to_year_incl) or df[COLUMN_WEEK].le(to_week))]

    # Add counts...or
    df[COLUMN_CASES] = df[COLUMN_WEEK].map(counts)

    # Sort chronologically...
    df[COLUMN_WEEK] = df['tmp']
    del df['tmp']
    df = df.sort_values(by=[COLUMN_YEAR, COLUMN_WEEK], ascending=[True, True])

    # Assign an unique number to each distinct combination of year and week
    def __make_id(data_frame):
        str_id = data_frame.apply(lambda x: '_'.join(map(str, x)), axis=1)
        return pd.factorize(str_id)[0] + 1

    df[COLUMN_WEEK] = __make_id(df[[COLUMN_YEAR, COLUMN_WEEK]])

    # Remove obsolete columns...
    del df[COLUMN_YEAR]

    log.info('Instances loaded successfully!')
    return df


def __get_attributes(df):
    attributes = []

    for column in df.columns:
        dtype = df[column].dtype

        if dtype == np.float64:
            attribute_type = 'REAL'
        elif dtype == np.int64:
            attribute_type = 'NUMERIC'
        else:
            attribute_type = list(df[column].value_counts(dropna=True).keys())

        attributes.append((column, attribute_type))

    return attributes


def __get_data(df):
    data = []

    for _, row in df.iterrows():
        data.append(row.tolist())

    return data


def __write_arff(output_file, dataset):
    with open(output_file, 'w') as f:
        arff.dump(dataset, f)


def create_arff(data_dir: str, dataset: str, feature_definition: str, from_year: int, from_week: int, to_year: int,
                to_week: int, output_file, count_file_name):
    if os.path.isfile(output_file):
        log.info('No need to create ARFF file \'' + str(output_file) + '\'. It does already exist.')
    else:
        counts = __load_counts(data_dir, count_file_name if count_file_name is not None else dataset + '_counts',
                               from_year, from_week, to_year, to_week)
        feature_names = __load_feature_names(data_dir, feature_definition)
        df = __load_instances(data_dir, dataset, from_year, from_week, to_year, to_week, counts, feature_names)
        log.info('Creating ARFF file \'' + str(output_file) + '\'...')
        arff_dataset = {
            'description': dataset,
            'relation': dataset,
            'attributes': __get_attributes(df),
            'data': __get_data(df)
        }
        __write_arff(output_file, arff_dataset)
        log.info('ARFF file created successfully!')


def __write_xml(output_file, labels):
    root_element = XmlTree.Element('labels')
    root_element.set('xmlns', 'http://mulan.sourceforge.net/labels')

    for label in labels:
        label_element = XmlTree.SubElement(root_element, 'label')
        label_element.set('name', label)

    with open(output_file, mode='w') as f:
        xml_string = minidom.parseString(XmlTree.tostring(root_element)).toprettyxml(encoding=ENCODING)
        f.write(xml_string.decode(ENCODING))


def create_xml(output_file):
    if os.path.isfile(output_file):
        log.info('No need to create XML file \'' + str(output_file) + '\'. It does already exist.')
    else:
        log.info('Creating XML file \'' + str(output_file) + '\'...')
        __write_xml(output_file, [COLUMN_WEEK, COLUMN_CASES])
        log.debug('XML file created successfully!')


class SyndromeLearnerRunnable(RuleLearnerRunnable):

    def _create_learner(self, args):
        return SyndromeLearner(from_year=args.from_year, from_week=args.from_week, to_year=args.to_year,
                               to_week=args.to_week, random_state=args.random_state, feature_format=args.feature_format,
                               max_rules=args.max_rules, time_limit=args.time_limit,
                               feature_sub_sampling=args.feature_sub_sampling, min_support=args.min_support,
                               max_conditions=args.max_conditions, num_threads_refinement=args.num_threads_refinement)

    def _preprocess(self, args) -> (str, str):
        log.info('Preprocessing raw data...')
        data_dir = args.data_dir
        dataset = args.dataset
        from_year = args.from_year
        from_week = args.from_week
        to_year = args.to_year
        to_week = args.to_week
        temp_dir = args.temp_dir
        count_file_name = args.count_file_name
        if temp_dir is None:
            raise ValueError('Mandatory parameter \'--temp-dir\' has not been specified')
        feature_definition = args.feature_definition
        if feature_definition is None:
            raise ValueError('Mandatory parameter \'--feature-definition\' has not been specified')

        output_file_name = dataset + '_' + feature_definition + '_' + str(from_year) + (
            '' if from_week < 0 else '-' + str(from_week)) + '_' + str(to_year) + (
                               '' if to_week < 0 else '-' + str(to_week))

        arff_file = os.path.join(temp_dir, output_file_name + '.arff')
        create_arff(data_dir, dataset, feature_definition, from_year, from_week, to_year, from_week, arff_file,
                    count_file_name)

        xml_file = os.path.join(temp_dir, output_file_name + '.xml')
        create_xml(xml_file)

        log.info('Raw data was preprocessed successfully!')
        return temp_dir, output_file_name


if __name__ == '__main__':
    parser = ArgumentParserBuilder(description='A multi-label classification experiment using BOOMER') \
        .add_time_series_learner_arguments() \
        .build()
    runnable = SyndromeLearnerRunnable()
    runnable.run(parser)
