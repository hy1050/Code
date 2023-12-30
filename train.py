import numpy as np
import joblib
from config import get_default_cfg
from model import TrafficSignRecognizer
import argparse


def accuracy(predictions, labels):
    print(np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)))
    print(prediction.shape[0])
    return (100 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            predictions.shape[0])


def load_dataset_and_labels(dataset_fname, train_or_test):
    data = joblib.load(dataset_fname)
    if train_or_test == 'train':
        dataset, labels = data['train_bboxes'], data['train_classIds']
    else:
        dataset, labels = data['test_bboxes'], data['test_classIds']
    return dataset, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='train_model_dir',
                        help='Path to model directory')
    args = parser.parse_args()

    config = get_default_cfg()

    # Load dataset and label
    train_dataset, train_labels = load_dataset_and_labels(
        config.TRAIN_PKL_FILENAME, 'train')
    test_dataset, test_labels = load_dataset_and_labels(
        config.TEST_PKL_FILENAME, 'test')

    # Create model and train
    model = TrafficSignRecognizer(mode='train', model_dir=args.model_dir)
    model.train(train_dataset, train_labels, learning_rate=1e-4)


if __name__ == '__main__':
    main()
