# Credit Card Fraud Sample

The code sample in this directory uses the
[Credit Card Fraud data set](https://github.com/IBM/TabFormer/tree/main/data/credit_card)
and deploys a saved model for TensorFlow Serving. Two scripts then perform
inference on the served model using gRPC and REST, and display the results.

The [tensorflow-serving README file](../../README.md) contains general
information on downloading and running the samples.

These samples require first downloading the data set from the Internet.

## Prerequisites

The sample depends on packages (`scikit-learn`, `pandas`, `joblib`,
`tensorflow-serving-api`) that are not included in the base IBM Z Accelerated
for TensorFlow container. `prerequisites.sh` handles this by building a new
container image on your behalf:

1. It passes your chosen IBM Z Accelerated for TensorFlow production image as a
   build argument to `Containerfile`.
2. `Containerfile` temporarily switches to `root` to run the `dnf` and `pip`
   installs, then drops back to `ibm-user` as the runtime user.
3. Once the image is built, `prerequisites.sh` prints the generated image tag
   and the `docker run` command to start the container. The sample scripts are
   mounted read-only at `/scripts` and the named volume is mounted at
   `/workspace` for all output files.

This container is used for **deployment and inference** only. The TensorFlow
Serving container is started separately as described below.

Before running `prerequisites.sh`, you must first train the model using the
[Credit Card Fraud training sample for the TensorFlow container](https://github.com/IBM/ibmz-accelerated-for-tensorflow/tree/main/samples/credit-card-fraud).

Run the script on the **host** (not from inside a container), passing your
IBM Z Accelerated for TensorFlow production image as the argument:

```bash
./prerequisites.sh <base-image>
```

For example:

```bash
./prerequisites.sh icr.io/ibmz/ibmz-accelerated-for-tensorflow:1.6.0
```

This builds a local image and prints the generated image tag (e.g.
`tensorflow-serving-ccf-sample:20250714-143022`) along with the `docker run`
command to start the container.

## Copying Training Artifacts into the Container

Once the container is running, open a second terminal on the host and copy the
training artifacts directly from the running TensorFlow CCF training container:

```bash
# Find the running container IDs
docker ps

# Copy the trained model and supporting files from the TF CCF training container
# To run against the GRU model, replace fitted_mapper_v2_lstm.pkl with fitted_mapper_v2_gru.pkl
docker cp <tf-ccf-training-container-id>:/workspace/saved_model <tf-serving-ccf-container-id>:/workspace/
docker cp <tf-ccf-training-container-id>:/workspace/fitted_mapper_v2_lstm.pkl <tf-serving-ccf-container-id>:/workspace/
docker cp <tf-ccf-training-container-id>:/workspace/test_100k.csv <tf-serving-ccf-container-id>:/workspace/
docker cp <tf-ccf-training-container-id>:/workspace/test_100k.indices <tf-serving-ccf-container-id>:/workspace/
```

Then return to the container shell to run the sample.

## Deploying the Model for TensorFlow Serving

From inside the container, run the `credit_card_fraud_deployment.py` script.
This exports the trained Keras model to a TensorFlow Serving Servable and
creates a warmup file.

**For the LSTM model (default):**

```bash
python /scripts/credit_card_fraud_deployment.py
```

**For the GRU model:**

```bash
python /scripts/credit_card_fraud_deployment.py --rnn-type gru
```

This creates a `serving_model/lstm/` or `serving_model/gru/` directory in
`/workspace`. Once complete, exit the container:

```bash
exit
```

## Serving the Model

Start the IBM Z Accelerated for TensorFlow Serving container, mounting the
workspace volume. The `--model_name` and `--model_base_path` must match the
model type you deployed above.

**For the LSTM model:**

```bash
docker run --rm --detach \
    -p 8500:8500 -p 8501:8501 \
    -v tensorflow-serving-ccf-workspace:/workspace \
    -e MODEL_NAME=lstm \
    --entrypoint tensorflow_model_server \
    icr.io/ibmz/ibmz-accelerated-serving-for-tensorflow:X.X.X \
    --port=8500 --rest_api_port=8501 \
    --model_name=lstm \
    --model_base_path=/workspace/serving_model/lstm
```

**For the GRU model:**

```bash
docker run --rm --detach \
    -p 8500:8500 -p 8501:8501 \
    -v tensorflow-serving-ccf-workspace:/workspace \
    -e MODEL_NAME=gru \
    --entrypoint tensorflow_model_server \
    icr.io/ibmz/ibmz-accelerated-serving-for-tensorflow:X.X.X \
    --port=8500 --rest_api_port=8501 \
    --model_name=gru \
    --model_base_path=/workspace/serving_model/gru
```

Replace `X.X.X` with the current version of the serving container image.

You can verify the model is ready with (substitute `gru` if applicable):

```bash
curl http://localhost:8501/v1/models/lstm/metadata
```

## Running Inference on the Served Model

Start a new container shell with `--network=host` so the inference scripts
can reach the serving container on `localhost`. Use the image tag printed by
`prerequisites.sh`:

```bash
docker run -it --rm \
    --entrypoint bash \
    --network=host \
    -v "$(pwd)":/scripts:ro,z \
    -v tensorflow-serving-ccf-workspace:/workspace \
    -w /workspace \
    tensorflow-serving-ccf-sample:<timestamp>
```

> Note: `--network=host` is used here for convenience in a sample environment.
> In production, use a dedicated container network instead.

From inside the container, run inference using gRPC:

**For the LSTM model (default):**

```bash
python /scripts/credit_card_fraud_grpc.py
```

**For the GRU model:**

```bash
python /scripts/credit_card_fraud_grpc.py --rnn-type gru
```

Or using REST:

**For the LSTM model (default):**

```bash
python /scripts/credit_card_fraud_rest.py
```

**For the GRU model:**

```bash
python /scripts/credit_card_fraud_rest.py --rnn-type gru
```

Both scripts report the test accuracy.

Once complete, exit the container:

```bash
exit
```

## Cleanup

When you are finished with the sample, stop the serving container, then remove
all containers, images, and the workspace volume:

```bash
docker stop $(docker ps -q --filter ancestor=icr.io/ibmz/ibmz-accelerated-serving-for-tensorflow:X.X.X)
docker container prune -f
docker rmi tensorflow-serving-ccf-sample:<timestamp>
docker rmi icr.io/ibmz/ibmz-accelerated-serving-for-tensorflow:X.X.X
docker volume rm tensorflow-serving-ccf-workspace
```

## Known Issues

There are no known open issues with this sample.
