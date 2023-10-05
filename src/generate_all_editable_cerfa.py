"""
Module allowing to generate samples for all cerfa templates present in data/CERFA/toFill
"""
import argparse

import math

from src.util.dataGeneration import Writer
from util.dataGeneration.baseWriter import Annotator, AnnotatorJson


def run_all(nb_samples=10, annotator:Annotator=AnnotatorJson(), train_test_split=False):
    """
    Generate samples for all cerfa templates present in data/CERFA/toFill
    :param int nb_samples: Number of pdf to generate
    :return: None
    """

    if train_test_split:
        nb_train = math.floor(0.8 * nb_samples)
        nb_test = nb_samples - nb_train
        nb_eval = int(0.2 * nb_train)
        nb_train = nb_train - nb_eval

        datasets = ["train"] * nb_train + ['validation'] * nb_eval + ['test'] * nb_test

    for subWriterClass in Writer.__subclasses__():
        subWriterClassName = subWriterClass.__name__
        num_cerfa = subWriterClassName.replace("Writer", '')

        for idx in range(nb_samples):
            writer = subWriterClass(num_cerfa=num_cerfa, annotator=annotator)
            writer.fill_form()
            writer.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--nb_samples', default=10, type=int,
                        help='The number of samples to generate for each cerfa')

    args = parser.parse_args()
    run_all(nb_samples = vars(args)['nb_samples'])

