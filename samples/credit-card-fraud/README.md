# Credit Card Fraud Sample

The code sample in this directory uses the
[Credit Card Fraud data set](https://github.com/IBM/TabFormer/tree/main/data/credit_card)
and deploys a saved model for TensorFlow Serving. Two scripts then perform
inference on the served model using gRPC and REST, and display the results.

The [tensorflow-serving README file](../../README.md) contains general
information on downloading and running the samples.

> If you are using rootless podman, see the
> [Running with podman](../README.md#running-with-podman) section in the
> top-level samples README before proceeding. The `prerequisites.sh` script
> handles the necessary podman setup automatically.

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
3. Once the image is built, `prerequisites.sh` creates a `workspace/`
   directory alongside the sample scripts, then starts an interactive shell
   inside the container with:
   - The sample scripts mounted read-only at `/scripts`
   - The `workspace/` directory mounted at `/workspace` (writable)

This container is used for **deployment and inference** only. The TensorFlow
Serving container is started separately as described below.

Before running `prerequisites.sh`, you must first train the model using the
[Credit Card Fraud training sample for the TensorFlow container](https://github.com/IBM/ibmz-accelerated-for-tensorflow/tree/main/samples/credit-card-fraud).
Once complete, copy the following files from that sample's `workspace/` into
this sample's `workspace/`:

- `saved_model/` directory (the trained Keras model)
- `fitted_mapper_v2_lstm.pkl` (or `fitted_mapper_v2_gru.pkl` for GRU)
- `test_100k.csv` and `test_100k.indices`

Run the script on the **host** (not from inside a container), passing your
IBM Z Accelerated for TensorFlow production image as the argument:

```bash
./prerequisites.sh <base-image> [/path/to/card_transaction.v1.csv]
```

For example:

```bash
./prerequisites.sh icr.io/ibmz/ibmz-accelerated-for-tensorflow:1.6.0
```

This builds a local image tagged `ccf-serving-sample:latest` and drops you
into an interactive shell at `/workspace` inside the container.

## Deploying the Model for TensorFlow Serving

From inside the container, run the `credit_card_fraud_deployment.py` script.
This exports the trained Keras model to a TensorFlow Serving Servable and
creates a warmup file.

```bash
python /scripts/credit_card_fraud_deployment.py
```

This creates a `serving_model/` directory in `/workspace`. To deploy the GRU
model instead:

```bash
python /scripts/credit_card_fraud_deployment.py --rnn-type gru
```

Once complete, exit the container:

```bash
exit
```

## Serving the Model

Start the IBM Z Accelerated for TensorFlow Serving container, mounting the
`serving_model/` directory from the workspace:

```bash
docker run --rm --detach \
    -p 8500:8500 -p 8501:8501 \
    -v "$(pwd)/workspace/serving_model/lstm:/models/lstm:z" \
    -e MODEL_NAME=lstm \
    icr.io/ibmz/ibmz-accelerated-serving-for-tensorflow:X.X.X
```

Replace `X.X.X` with the current version of the serving container image. For
the GRU model, substitute `lstm` with `gru` in both the path and `MODEL_NAME`.

You can verify the model is ready with:

```bash
curl http://localhost:8501/v1/models/lstm/metadata
```

## Running Inference on the Served Model

Run these commands from the `credit-card-fraud/` sample directory. Start a new
container shell with `--network=host` so the inference scripts can reach the
serving container on `localhost`:

```bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
docker run -it --rm \
    --entrypoint bash \
    --network=host \
    -v "${SCRIPT_DIR}":/scripts:ro,z \
    -v "${SCRIPT_DIR}/workspace":/workspace:z \
    -w /workspace \
    ccf-serving-sample:latest
```

> Note: `--network=host` is used here for convenience in a sample environment.
> In production, use a dedicated container network instead.

From inside the container, run inference using gRPC:

```bash
python /scripts/credit_card_fraud_grpc.py
```

Or using REST:

```bash
python /scripts/credit_card_fraud_rest.py
```

Both scripts report the test accuracy. To run against the GRU model, add
`--rnn-type gru` to either command.

Once complete, exit the container:

```bash
exit
```

## Known Issues

There are no known open issues with this sample.
