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

import json
import tensorflow as tf

from six.moves import urllib

(X_train, y_train), (X_test, y_test) = \
    tf.keras.datasets.fashion_mnist.load_data()

X_test = X_test.astype('float32') / 255

# Reshape from [N, 28, 28] to expected input shape [N, 28, 28, 1]
X_test = tf.expand_dims(X_test, axis=-1)

# Process request.
req = {"signature_name": "serving_default",
       "inputs": X_test.numpy().tolist()}
data = json.dumps(req).encode('utf-8')
url = 'http://localhost:8501/v1/models/fashion_mnist:predict'

try:
    resp = urllib.request.urlopen(urllib.request.Request(url, data=data))
    resp_data = resp.read()
    resp.close()
except Exception as e:  # pylint: disable=broad-except
    print('Request failed. Error: {}'.format(e))
    raise e

# Process output.
output = json.loads(resp_data.decode())['outputs']
y_pred = tf.convert_to_tensor(output, dtype=tf.float32)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_test, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)
print('Test accuracy:', accuracy.numpy())

