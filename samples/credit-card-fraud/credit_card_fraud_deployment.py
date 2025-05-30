#!/usr/bin/env python3

# IBM Confidential
# © Copyright IBM Corp. 2025

"""
Credit Card Fraud Deployment
"""

import argparse
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf

from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2

import data_utils


def prepare_model(rnn_type: str = 'lstm') -> tf.keras.models.Model:
    """
    Setup to get the model. Return the compiled model.
    """

    keras_model_path = f'./saved_model/{rnn_type}.keras'
    model = tf.keras.models.load_model(keras_model_path)

    return model


def main(rnn_type: str = 'lstm', batch_size: int = 2000, seq_length: int = 7):
    """
    main
    """

    test_generator = data_utils.prepare_inference_data(batch_size, seq_length)

    model = prepare_model(rnn_type)

    # Export model to a SavedModel Servable for TensorFlow Serving.
    if not os.path.exists('./serving_model'):
        os.makedirs('./serving_model')
    version = 1
    serving_model_path = f'./serving_model/{rnn_type}/{version}/'
    model.export(serving_model_path)

    # Create a PredictionLog file and add it to the Servable to be used as a
    # SavedModel Warmup.
    warmup_path = serving_model_path + "assets.extra/"
    if not os.path.exists(warmup_path):
        os.mkdir(warmup_path)
    warmup_file = warmup_path + "tf_serving_warmup_requests"

    input_batch, _ = next(test_generator)
    tensors = {}
    for p in model.inputs:
        tensors[p.name] = tf.constant(input_batch, dtype=tf.float32)
    example_inputs = {k: tf.make_tensor_proto(v) for k, v in tensors.items()}

    with tf.io.TFRecordWriter(warmup_file) as writer:
        request = predict_pb2.PredictRequest(
            model_spec=model_pb2.ModelSpec(name=model.name),
            inputs=example_inputs
        )
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request)
        )
        writer.write(log.SerializeToString())


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
