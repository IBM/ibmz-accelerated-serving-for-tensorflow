# Copyright 2023 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import grpc
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

(X_train, y_train), (X_test, y_test) = \
    tf.keras.datasets.fashion_mnist.load_data()

X_test = X_test.astype('float32') / 255

# Reshape from [N, 28, 28] to expected input shape [N, 28, 28, 1]
X_test = tf.expand_dims(X_test, axis=-1)

# Process request.
request = predict_pb2.PredictRequest()
request.model_spec.name = "fashion_mnist"
request.model_spec.signature_name = "serving_default"

# Process input.
request.inputs["conv2d_input"].CopyFrom(tf.make_tensor_proto(X_test,
                                                             dtype=tf.float32))
request.output_filter.append("dense_2")

# Send request.
channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
outputs = stub.Predict(request, 5.0)

# Process output.
y_pred = tf.make_ndarray(outputs.outputs["dense_2"])

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_test, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)
print('Test accuracy:', accuracy.numpy())

