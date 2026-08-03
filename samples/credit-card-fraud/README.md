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
3. Once the image is built, `prerequisites.sh` starts an interactive shell
   inside the container with a named volume mounted at `/workspace` (writable).

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
`tensorflow-serving-ccf-sample:20250714-143022`) along with the `docker run` command
to start the container.

## Copying Scripts and Training Artifacts into the Container

Once the container is running, open a second terminal on the host and use
`docker cp` to copy the sample scripts into the container:

```bash
# Find the running container ID
docker ps

# Copy the sample scripts
docker cp credit_card_fraud_deployment.py <container-id>:/workspace/
docker cp credit_card_fraud_grpc.py <container-id>:/workspace/
docker cp credit_card_fraud_rest.py <container-id>:/workspace/
```

Also copy the training artifacts produced by the TensorFlow CCF training
sample into the container:

```bash
# Copy the trained model and supporting files
docker cp /path/to/saved_model <container-id>:/workspace/
docker cp /path/to/fitted_mapper_v2_lstm.pkl <container-id>:/workspace/
docker cp /path/to/test_100k.csv <container-id>:/workspace/
docker cp /path/to/test_100k.indices <container-id>:/workspace/
```

Then return to the container shell to run the sample.

## Deploying the Model for TensorFlow Serving

From inside the container, run the `credit_card_fraud_deployment.py` script.
This exports the trained Keras model to a TensorFlow Serving Servable and
creates a warmup file.

```bash
python credit_card_fraud_deployment.py
```

This creates a `serving_model/` directory in `/workspace`. To deploy the GRU
model instead:

```bash
python credit_card_fraud_deployment.py --rnn-type gru
```

Once complete, exit the container:

```bash
exit
```

## Serving the Model

Start the IBM Z Accelerated for TensorFlow Serving container, mounting the
`serving_model/` directory from the workspace volume:

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

Replace `X.X.X` with the current version of the serving container image. For
the GRU model, substitute `lstm` with `gru` in the `--model_name` and
`--model_base_path` arguments.

You can verify the model is ready with:

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
    -v tensorflow-serving-ccf-workspace:/workspace \
    -w /workspace \
    tensorflow-serving-ccf-sample:<timestamp>
```

> Note: `--network=host` is used here for convenience in a sample environment.
> In production, use a dedicated container network instead.

From inside the container, run inference using gRPC:

```bash
python credit_card_fraud_grpc.py
```

Or using REST:

```bash
python credit_card_fraud_rest.py
```

Both scripts report the test accuracy. To run against the GRU model, add
`--rnn-type gru` to either command.

Once complete, exit the container:

```bash
exit
```

## Known Issues

There are no known open issues with this sample.
