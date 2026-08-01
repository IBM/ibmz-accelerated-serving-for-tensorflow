#!/usr/bin/env python3

# IBM Confidential
# © Copyright IBM Corp. 2025, 2026

"""
Credit Card Fraud gRPC Inference
"""

import argparse
from collections.abc import Generator
import math
from pathlib import Path
from typing import Any

import grpc
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


SEQ_LENGTH = 7


def time_encoder(x: pd.DataFrame) -> pd.DataFrame:
    """
    Encoder for time data.
    """

    x_hm = x['Time'].str.split(':', expand=True)
    x_date = pd.DataFrame({
        'year': x['Year'],
        'month': x['Month'],
        'day': x['Day'],
        'hour': x_hm[0],
        'minute': x_hm[1]})
    d = pd.to_datetime(x_date).astype(np.int64)
    return pd.DataFrame(d, columns=['Year_Month_Day_Time'])


def amt_encoder(x: pd.DataFrame) -> pd.DataFrame:
    """
    Encoder for decimal data.
    """

    return x.map(lambda amt: amt.lstrip('$')).astype(np.float32).map(
        lambda amt: max(1.0, amt)).map(math.log)


def decimal_encoder(x: pd.DataFrame, length: int = 5) -> pd.DataFrame:
    """
    Encoder for integer data.
    """

    col_name = x.columns[0]
    x = np.ravel(x)
    x_new = pd.DataFrame()
    for i in range(length):
        x_new[f'{col_name}_x{i}'] = np.mod(x, 10)
        x = np.floor_divide(x, 10)
    return x_new.astype(np.int64)


def fraud_encoder(x: pd.DataFrame) -> pd.DataFrame:
    """
    Encoder for boolean data.
    """

    return x.map(lambda v: '1' if v == 'Yes' else '0').astype(np.int64)


def gen_inference_batch(
        df: pd.DataFrame, mapper: ColumnTransformer,
        indices: np.ndarray[np.int64, Any],
        batch_size: int) -> Generator[tuple[np.ndarray, np.ndarray], Any, Any]:
    """
    Generator that yields batches with shape:
        data    = [batch_size, SEQ_LENGTH, features]
        targets = [batch_size, 1]
    """

    rows = indices.shape[0]
    index_array = np.zeros((rows, SEQ_LENGTH), dtype=np.int64)
    for i in range(SEQ_LENGTH):
        index_array[:, i] = indices + 1 - SEQ_LENGTH + i

    count = 0
    while count < rows:
        end = min(count + batch_size, rows)
        batch_rows = end - count
        batch_df = df.loc[index_array[count:end].flatten()]
        batch_df = batch_df.reset_index(drop=True)
        batch_df = mapper.transform(batch_df)
        batch_data = batch_df.drop(
            ['Is Fraud?'], axis=1).to_numpy().reshape(batch_rows, SEQ_LENGTH, -1)
        batch_targets = batch_df['Is Fraud?'].to_numpy().reshape(
            batch_rows, SEQ_LENGTH, 1)
        # Take the label for the final sample in sequence as the sequence label.
        batch_targets = batch_targets[:, -1, :]
        count = end
        yield batch_data, batch_targets


def prepare_inference_data(
        rnn_type: str, batch_size: int, seq_length: int) \
        -> Generator[tuple[np.ndarray, np.ndarray], Any, Any]:
    """
    Load and preprocess inference data.
    """

    csv_path = Path('./test_100k.csv')
    x_original = pd.read_csv(csv_path, index_col='Index')

    indices_path = Path('./test_100k.indices')
    test_indices = np.loadtxt(indices_path).astype(np.int64)

    mapper_path = f'./fitted_mapper_v2_{rnn_type}.pkl'
    print('Loading saved mapper . . .')
    with open(mapper_path, 'rb') as f:
        fitted_mapper = joblib.load(f)

    return gen_inference_batch(x_original, fitted_mapper, test_indices, batch_size)


def main(rnn_type: str = 'lstm', batch_size: int = 2048, seq_length: int = 7):
    """
    main
    """

    test_generator = prepare_inference_data(rnn_type, batch_size, seq_length)

    y_pred = []
    y_true = []

    for input_batch, label_batch in test_generator:
        # Process request.
        request = predict_pb2.PredictRequest()
        request.model_spec.name = rnn_type
        request.model_spec.signature_name = "serving_default"

        # Process input.
        request.inputs["input_layer"].CopyFrom(
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
        default=2048,
        help='Batch size for inference (default: 2048)',
    )
    parser.add_argument(
        '--seq-length',
        type=int,
        default=7,
        help='Sequence length (default: 7)',
    )
    args = parser.parse_args()

    main(args.rnn_type, args.batch_size, args.seq_length)
