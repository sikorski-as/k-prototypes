import argparse
import os
import re

from clustering import cluster, ClusteringError


def column_specification(argument_value):
    specs = argument_value.split(',')
    pattern = re.compile(r'[0-9]+((?::)[0-9]+)?(?:,|$)')
    col_ids = set()
    for spec in specs:
        matched = pattern.match(spec)
        if not matched:
            raise argparse.ArgumentTypeError('invalid format')
        spec = tuple(int(str_number) for str_number in matched.group().split(':'))
        if len(spec) == 1:
            col_ids.add(spec[0])
        elif len(spec) == 2:
            col_ids.update(range(spec[0], spec[1] + 1))
        else:
            raise argparse.ArgumentTypeError('invalid format')

    return list(col_ids)


def filepath(arg):
    if not os.path.exists(arg):
        raise argparse.ArgumentTypeError('file does not exist')
    else:
        return arg


def int_greater_than(value):
    def greater_than_(value_):
        ivalue = int(value_)
        if ivalue <= value:
            raise argparse.ArgumentTypeError('value has to be greater than {}'.format(value))
        return ivalue

    return greater_than_


def main():
    parser = argparse.ArgumentParser(description='K-prototypes algorithm program.')
    parser.add_argument('datafile',
                        type=filepath,
                        help='CSV data file to be used')
    parser.add_argument('-print-report', '--print-report',
                        action='store_true',
                        help='should report be printed on the screen?')
    parser.add_argument('-num', '--numerical',
                        type=column_specification,
                        default=[],
                        help='comma separated list of columns to be used as numerical attributes for clustering'
                             'eg.: 1,2,3,4,5 or 1:5 or 1:4,5 or 1:2, 3, 4:5')
    parser.add_argument('-nom', '--nominal',
                        type=column_specification,
                        default=[],
                        help='comma separated list of columns to be used as nominal attributes for clustering, '
                             'eg.: 1,2,3,4,5 or 1:5 or 1:4,5 or 1:2, 3, 4:5')
    parser.add_argument('-gs', '--true-labels',
                        type=int_greater_than(-1),
                        default=None,
                        help='column id with true cluster labels (gold standard)')
    parser.add_argument('k',
                        type=int_greater_than(1),
                        help='number of expected clusters')
    parser.add_argument('--attempts',
                        type=int_greater_than(0),
                        default=1,
                        help='how many time clustering should be repeated to find the best model')
    parser.add_argument('-a', '--alpha',
                        type=float,
                        default=1.0,
                        help='scaling factor for numerical distance')
    parser.add_argument('-b', '--beta',
                        type=float,
                        default=1.0,
                        help='scaling factor for nominal distance')
    parser.add_argument('-ss', '--standardize-std',
                        action='store_true',
                        help='scale numerical features with std')
    parser.add_argument('-sm', '--standardize-mean',
                        action='store_true',
                        help='center numerical features with mean')
    args = parser.parse_args()
    if args.numerical == [] and args.nominal == []:
        parser.error("at least one of --numerical and --nominal required")

    try:
        report = cluster(args.datafile, args.k,
                         args.numerical, args.nominal, args.true_labels,
                         args.alpha, args.beta,
                         args.standardize_mean, args.standardize_std,
                         args.attempts)
        if args.print_report:
            print(report)
    except ClusteringError as e:
        print('Clustering error:', e)
        exit(1)


if __name__ == '__main__':
    main()
