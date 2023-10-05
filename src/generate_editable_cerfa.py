"""
Module allowing to generate samples for a given cerfa
"""
import argparse
import math
import os

from src.util.dataGeneration import Writer
from src.util.dataGeneration.baseWriter import Annotator, AnnotatorTxt, AnnotatorJson, AnnotatorJsonDonut
from util.utils import pdf_to_image


def run(num_cerfa, annotator:Annotator, nb_samples=10, train_test_split=False):
    """
    Generate samples for a given cerfa
    :param str num_cerfa: Cerfa number we want to generate (The empty template and the config json file must be
    present in data/CERFA/toFill)
    :param int nb_samples: Number of pdf to generate
    :return: None
    """
    sub_writers = Writer.__subclasses__()
    matching_writers = [w for w in sub_writers if num_cerfa in w.__name__]
    if len(matching_writers)>1:
        raise ValueError(f"Ambiguous cerfa number: {num_cerfa}, multiple writers are matching:\n"
                         f"{[w.__name__ for w in matching_writers]}")
    elif len(matching_writers)==1:
        custom_writer = matching_writers[0]

        if train_test_split:
            nb_train = math.floor(0.8*nb_samples)
            nb_test = nb_samples - nb_train
            nb_eval = int(0.2*nb_train)
            nb_train = nb_train - nb_eval

            datasets = ["train"]*nb_train + ['validation']*nb_eval + ['test']*nb_test

        for idx in range(nb_samples):
            writer = custom_writer( num_cerfa = num_cerfa, annotator=annotator)
            writer.fill_form()

            if train_test_split:
                writer.save_by_dataset(datasets[idx])
            else:
                writer.save()

        if train_test_split:
            for dataset in set(datasets):
                out_pdf = writer.output_filepath.format(num_cerfa,
                                                      dataset,
                                                      "DELETE_ME")
                in_folder = os.sep.join(out_pdf.split("DELETE_ME")[:-1])
                out_folder = in_folder
                pdf_to_image(in_folder, out_folder)

    else:
        raise ValueError(f"No custom writer found for cerfa : {num_cerfa}. Available writers are:\n"
                         f"{[w.__name__ for w in sub_writers]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('num_cerfa', type=str, nargs=1,
                        help='The cerfa number')
    parser.add_argument('annotator', type=str, nargs=1,
                        help='Annotation type (Must be in : [AnnotatorTxt, AnnotatorJson, AnnotatorJsonDonut] )')
    parser.add_argument('--nb_samples', default=10, type=int,
                        help='The number of samples to generate for each cerfa')
    parser.add_argument('--train_test_split', default=False, type=bool,
                        help='Weather to split forms into train/eval/test datasets')

    args = parser.parse_args()

    run(num_cerfa = args.num_cerfa[0],
        annotator =  eval(args.annotator[0])(),
        nb_samples = vars(args)['nb_samples'],
        train_test_split = vars(args)['train_test_split'])

