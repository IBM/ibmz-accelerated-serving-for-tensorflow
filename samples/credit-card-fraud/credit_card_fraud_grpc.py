#!/usr/bin/env python3

# IBM Confidential
# © Copyright IBM Corp. 2025

"""
Credit Card Fraud Inference
"""

import argparse

import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

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
        request = predict_pb2.PredictRequest()
        request.model_spec.name = rnn_type
        request.model_spec.signature_name = "serving_default"

        # Process input.
        request.inputs["input_1"].CopyFrom(
            tf.make_tensor_proto(input_batch, dtype=tf.float32))
        request.output_filter.append("output_0")

        # Send request.
        channel = grpc.insecure_channel('localhost:8500')
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        outputs = stub.Predict(request, 5.0)

        # Process output.
        y_pred.extend(tf.make_ndarray(outputs.outputs["output_0"]))
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
