#!/usr/bin/env python3

# IBM Confidential
# © Copyright IBM Corp. 2025, 2026

"""
Credit Card Fraud Deployment
"""

import argparse
from collections.abc import Generator
import math
import os
from pathlib import Path
from typing import Any

os.environ["KERAS_BACKEND"] = "tensorflow"

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
import tensorflow as tf

from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2


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


def prepare_model(rnn_type: str) -> tf.keras.models.Model:
    """
    Load the saved model.
    """

    keras_model_path = f'./saved_model/{rnn_type}.keras'
    return tf.keras.models.load_model(keras_model_path)


def main(rnn_type: str = 'lstm', batch_size: int = 2048, seq_length: int = 7):
    """
    main
    """

    test_generator = prepare_inference_data(rnn_type, batch_size, seq_length)

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
