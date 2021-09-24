project('tsa', 'cpp', version : '0.1.0')

# Source files
source_files = [
    'src/tsa/data/matrix_dense_numeric.cpp',
    'src/tsa/model/rule_list.cpp',
    'src/tsa/rule_evaluation/rule_evaluation_label_wise_regularized.cpp',
    'src/tsa/statistics/statistics_label_wise_dense.cpp',
    'src/tsa/statistics/statistics_label_wise_provider.cpp'
]

# Dependencies
common_dep = subproject('common').get_variable('common_dep')
openmp_dep = dependency('openmp')

dependencies = [
    common_dep,
    openmp_dep
]

# Directory containing public headers
include_directories = include_directories('include')

tsa_lib = library('rltsa', source_files, include_directories : include_directories, dependencies : dependencies,
                  version : '0.1.0')
