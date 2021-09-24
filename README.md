# README

This project provides a rule learning algorithm for the deduction of syndrome definitions from time series data. Large parts of the algorithm are based on ["BOOMER"](https://github.com/mrapp-ke/Boomer). 

## Features

The algorithm that is provided by this project currently supports the following functionalities for learning descriptive rules:

* The quality of rules is assessed by comparing the predictions of the current model to the ground truth in terms of the Pearson correlation coefficient.
* When learning a new rule, random samples of the features may be used.
* Hyper-parameters that provide control over the specificity/generality of rules are available.
* The algorithm can natively handle numerical, ordinal and nominal features (without the need for pre-processing techniques such as one-hot encoding).
* The algorithm is able to deal with missing feature values, i.e., occurrences of NaN in the feature matrix.

In addition, the following features that may speed up training or reduce the memory footprint are currently implemented:

* Dense or sparse feature matrices can be used for training. The use of sparse matrices may speed-up training significantly on some data sets.
* Multi-threading can be used to parallelize the evaluation of a rule's potential refinements across multiple CPU cores. 

## Project structure

```
|-- cpp                     Contains the implementation of core algorithms in C++
    |-- subprojects
        |-- common          Contains implementations that all algorithms have in common
        |-- tsa             Contains implementations for time series analysis
    |-- ...
|-- python                  Contains Python code for running experiments
    |-- rl
        |-- common          Contains Python code that is needed to run any kind of algorithms
            |-- cython      Contains commonly used Cython wrappers
            |-- ...
        |-- tsa             Contains Python code for time series analysis
            |-- cython      Contains time series-specific Cython wrappers
            |-- ...
        |-- testbed         Contains useful functionality for running experiments
            |-- ...
    |-- main.py             Can be used to start an experiment
    |-- ...
|-- Makefile                Makefile for compilation
|-- ...
```

## Project setup

The algorithm provided by this project is implemented in C++. In addition, a Python wrapper that implements the scikit-learn API is available. To be able to integrate the underlying C++ implementation with Python, [Cython](https://cython.org) is used.

The C++ implementation, as well as the Cython wrappers, must be compiled in order to be able to run the provided algorithm. To facilitate compilation, this project comes with a Makefile that automatically executes the necessary steps.

At first, a virtual Python environment can be created via the following command:
```
make venv
```

As a prerequisite, Python 3.7 (or a more recent version) must be available on the host system. All compile-time dependencies (`numpy`, `scipy`, `Cython`, `meson` and `ninja`) that are required for building the project will automatically be installed into the virtual environment. As a result of executing the above command, a subdirectory `venv` should have been created within the project's root directory.

Afterwards, the compilation can be started by executing the following command:
```
make compile
```

Finally, the library must be installed into the virtual environment, together with all of its runtime dependencies (e.g. `scikit-learn`, a full list can be found in `setup.py`). For this purpose, the project's Makefile provides the following command:

```
make install
```

*Whenever any C++ or Cython source files have been modified, they must be recompiled by running the command `make compile` again! If compilation files do already exist, only the modified files will be recompiled.*

**Cleanup:** To get rid of any compilation files, as well as of the virtual environment, the following command can be used:
```
make clean
``` 

For more fine-grained control, the command `make clean_venv` (for deleting the virtual environment) or `make clean_compile` (for deleting the compiled files) can be used. If only the compiled Cython files should be removed, the command `make clean_cython` can be used. Accordingly, the command `make clean_cpp` removes the compiled C++ files.

## Parameters

The file `python/main.py` allows to run experiments on a specific data set using different configurations of the learning algorithm. The implementation takes care of writing the experimental results into `.csv` files and the learned model can (optionally) be stored on disk to reuse it later. 

In order to run an experiment, the following command line arguments must be provided (most of them are optional):

| Parameter                  | Optional? | Default  | Description                                                                                                                                                                                                                                                  |
|----------------------------|-----------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--data-dir`               | No        | `None`   | The path of the directory where the data sets are located.                                                                                                                                                                                                   |
| `--temp-dir`               | No        | `None`   | The path of the directory where temporary files should be saved.                                                                                                                                                                                             |
| `--dataset`                | No        | `None`   | The name of the `.csv` files that store the raw data (without suffix).                                                                                                                                                                                       |
| `--feature-definition`     | No        | `None`   | The name of the `.txt` file that specifies the names of the features to be used (without suffix).                                                                                                                                                            |                             
| `--from-year`              | No        | `None`   | The first year (inclusive) that should be taken into account.                                                                                                                                                                                                |
| `--to-year`                | No        | `None`   | The last year (inclusive) that should be taken into account.                                                                                                                                                                                                 |
| `--from-week`              | Yes       | `-1`     | The first week (inclusive) of the first year that should be taken into account or `-1`, if all weeks of that year should be used.                                                                                                                            |  
| `--to-week`                | Yes       | `-1`     | The last week (inclusive) of the last year that should be taken into account or `-1`, if all weeks of that year should be used.                                                                                                                              |
| `--count-file-name`        | Yes       | `None`   | The name of the file that stores the number of cases that correspond to individual weeks (without suffix). If not specified, the results from appending "_counts" to the dataset name.                                                                       |
| `--one-hot-encoding`       | Yes       | `False`  | `True`, if one-hot-encoding should be used for nominal attributes, `False` otherwise.                                                                                                                                                                        |
| `--output-dir`             | Yes       | `None`   | The path of the directory into which the experimental results (`.csv` files) should be written.                                                                                                                                                              |
| `--print-rules`            | Yes       | `True`   | `True`, if the induced rules should be printed on the console, `False` otherwise.                                                                                                                                                                            |
| `--store-rules`            | Yes       | `True`   | `True`, if the induced rules should be stored as a `.txt` file, `False` otherwise. Does only have an effect if the parameter `--output-dir` is specified.                                                                                                    |
| `--print-options`          | Yes       | `{}`     | A dictionary that specifies additional options to be used for printing or storing rules, if the parameter `--print-rules` and/or `--store-rules` is set to `True`, e.g. `{'print_feature_names':True,'print_label_names':True,'print_nominal_values':True}`. |
| `--store-predictions`      | Yes       | `True`   | `True`, if the predictions for the training data should be stored as a `.csv` file, `False` otherwise. Does only have an effect if the parameter `--output-dir` is specified.                                                                                |
| `--model-dir`              | Yes       | `None`   | The path of the directory where models (`.model` files) are located.                                                                                                                                                                                         |
| `--max-rules`              | Yes       | `50`     | The maximum number of rules to be induced or `-1`, if the number of rules should not be restricted.                                                                                                                                                          |
| `--time-limit`             | Yes       | `-1`     | The duration in seconds after which the induction of rules should be canceled or `-1`, if no time limit should be used.                                                                                                                                      |
| `--feature-sub-sampling`   | Yes       | `None`   | The name of the strategy to be used for feature sub-sampling. Must be `random-feature-selection` or `None`. Additional arguments may be provided as a dictionary, e.g. `random_feature-selection{'sample_size':0.5}`.                                        |
| `--min-support`            | Yes       | `0.0001` | The percentage of training examples that must be covered by a rule. Must be greater than `0` and smaller than `1`.                                                                                                                                           |
| `--max-conditions`         | Yes       | `-1`     | The maximum number of conditions to be included in a rule's body. Must be at least `1` or `-1`, if the number of conditions should not be restricted.                                                                                                        |
| `--random-state`           | Yes       | `1`      | The seed to the be used by random number generators.                                                                                                                                                                                                         |
| `--feature-format`         | Yes       | `auto`   | The format to be used for the feature matrix. Must be `sparse`, if a sparse matrix should be used, `dense`, if a dense matrix should be used, or `auto`, if the format should be chosen automatically.                                                       |
| `--num-threads-refinement` | Yes       | `1`      | The number of threads to be used to search for potential refinements of rules. Must be at least `1` or `-1`, if the number of cores that are available on the machine should be used.                                                                        |
| `--log-level`              | Yes       | `info`   | The log level to be used. Must be `debug`, `info`, `warn`, `warning`, `error`, `critical`, `fatal` or `notset`.                                                                                                                                              |


## Example and data format

In the following, we give a more detailed description of the data that must be provided to the algorithm. All input files must use UTF-8 encoding and they must be available in a single directory. The path of the directory must be specified via the parameter `--data-dir`. The following files must be included in the directory:

* A `.csv` file that stores the raw training data (see `data/example.csv` for an example). Each row (separated by line breaks) must correspond to an individual instance and the columns (separated by commas) must correspond to the available features. The names of the columns/features must be given as the first row. The names of columns can be arbitrary, but there must be a column named "week" that associates each instance with a corresponding year and week (using the format `year-month`, e.g. `2019-2`).
* A `.csv` file that specifies the number of cases that correspond to individual weeks (see `data/example_counts.csv` for an example). The file must consist of three columns, `year,week,cases`, separated by commas. The names of columns must be given as the first row. Each of the other rows (separated by line breaks) assigns a specific number of cases to a certain week of a year (all values must be positive integers). For each combination of year and week that occurs in the column "week" of the first `.csv` file, the number of cases must be specified in this second `.csv` file. 
* A `.txt` file that specifies the names of the features that should be taken into account (see `data/features.txt` for an example). Each feature name must be given as a new line. For each feature that is specified in the text file, a column with the same name must exist in the first `.csv` file.

The parameter `--dataset` is used to identify the `.csv` files that should be used by the algorithm. Its value must correspond to the name of the first `.csv` file mentioned above, omitting the file's suffix (e.g. `example` if the file's name is `example.csv`). The second `.csv` file must be named accordingly by appending the suffix `_counts` to the name of the first file (e.g. `example_counts.csv`).  The parameter `--feature-definition` is used to specify the name of the text file that stores the names of relevant features. The given value must correspond to the name of the text file, again omitting the file's suffix (e.g. `features`, if the file's name is `features.txt`).

In the following, the command for running an experiment, including all mandatory parameters, can be seen: 

```
venv/bin/python3 python/main.py --data-dir /path/to/data/ --temp-dir /path/to/temp/ --dataset example --feature-definition features --from-year 2018 --to-year 2019
```

When running the program for the first time, the `.csv` files that are located in the specified data directory will be loaded. The data will be filtered according to the parameters `--from-year` and `--to-year`, such that only instances that belong to the specified timespan are retained. Furthermore, all columns that are missing from the supplied text file will be removed. Finally, the data is converted into the format that is required for learning a rule model. This results in two files (an `.arff` file and a `.xml` file) that are written to the directory that is specified via the parameter `--temp-dir`. The resulting files are named according to the following scheme: `<dataset>_<feature-definition>_<from-year>-<to-year>` (e.g., `example_features_2018-2019`.) When running the program multiple times, it will check if the files do already exist. If this is the case, the preprocessing step will be skipped and the available files will be used as they are. 
