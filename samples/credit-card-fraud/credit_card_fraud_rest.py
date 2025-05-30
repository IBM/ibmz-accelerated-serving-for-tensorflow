#!/usr/bin/env python3

# IBM Confidential
# © Copyright IBM Corp. 2025

"""
Credit Card Fraud Inference
"""

import argparse
import json

import tensorflow as tf
from six.moves import urllib

import data_utils


def main(rnn_type: str = 'lstm', batch_size: int = 2000, seq_length: int = 7):
    """
    main
    """

    test_generator = data_utils.prepare_inference_data(batch_size, seq_length)

    y_pred = []
    y_true = []

    for input_batch, label_batch in test_generator:
        # Process request.
        req = {"signature_name": "serving_default",
               "inputs": input_batch.tolist()}
        data = json.dumps(req).encode('utf-8')
        url = f'http://localhost:8501/v1/models/{rnn_type}:predict'

        try:
            resp = urllib.request.urlopen(
                urllib.request.Request(url, data=data))
            resp_data = resp.read()
            resp.close()
        except Exception as e:  # pylint: disable=broad-except
            print('Request failed. Error: {}'.format(e))
            raise e

        # Process output.
        y_pred.extend(json.loads(resp_data.decode())['outputs'])
        y_true.extend(label_batch)

    y_pred = tf.concat(y_pred, axis=0)
    y_true = tf.constant(y_true)
    correct_prediction = tf.equal(tf.cast(tf.round(y_pred), tf.int32), y_true)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Test accuracy:', accuracy.numpy())


if __name__ == '__main__':
    # CLI interface
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rnn-type',
        type=str.lower,
        choices=['lstm', 'gru'],
        default='lstm',
        help='RNN type used within model (default: lstm)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2000,
        help='Batch size for training data (default: 2000)',
    )
    parser.add_argument(
        '--seq-length',
        type=int,
        default=7,
        help='Sequence length for training data (default: 7)',
    )
    args = parser.parse_args()

    main(args.rnn_type, args.batch_size, args.seq_length)