import argparse
import matplotlib.pyplot as plt
from visualise.performance_over_time import plot_median_values
from visualise.speedups_over_time import plot_speedups_over_time
from visualise.speedups import plot_speedups_for_file
import os


def get_parser() -> argparse.ArgumentParser:
    """
    Get the directory specified as an argument on the command line.

    Returns
    -------
    str
        The path of the directory
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-st", "--speedup-over-time", action="store",
                        dest="speedup_over_time_dir",
                        help="Plot and show how the speedups data has changed"
                             " over time for the files in the directory you"
                             " have specified as part of this argument.")
    parser.add_argument("-p", "--performance", action="store",
                        dest="performance_dir",
                        help="Plot and show how performance data has changed"
                             " over time for the files in the directory you"
                             " have specified as part of this argument")
    parser.add_argument("-sf", "--speedup-file", action="store",
                        dest="speedup_file",
                        help="Plot and show how using more threads affects the"
                             " performance of functions across multiple"
                             " different materials for the specified file.")
    return parser


def call_plot(dir_or_file: str, plot_func, plot_args):
    if os.path.isdir(dir_or_file) or os.path.isfile(dir_or_file):
        plot_func(*plot_args)
    else:
        print(
            "{} is not a recognised file or directory. "
            "If this is a network address you may"
            " not have access to it.".format(dir_or_file)
        )


if __name__ == "__main__":
    parser = get_parser()
    args_parsed = parser.parse_args()
    if args_parsed.speedup_over_time_dir:
        call_plot(
            args_parsed.speedup_over_time_dir,
            plot_speedups_over_time, args_parsed.speedup_over_time_dir
        )
    if args_parsed.performance_dir:
        call_plot(
            args_parsed.performance_dir,
            plot_median_values, args_parsed.performance_dir
        )
    if args_parsed.speedup_file:
        call_plot(
            args_parsed.speedup_file,
            plot_speedups_for_file, args_parsed.speedup_file
        )
    plt.show()
