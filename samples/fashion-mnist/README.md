# Fashion MNIST Serving Sample

The code sample in this directory trains a model on the
[Fashion MNIST data set](https://www.tensorflow.org/datasets/catalog/fashion_mnist)
and serves it using IBM Z Accelerated for TensorFlow Serving. Two inference
scripts exercise the served model over gRPC and REST.

The [tensorflow serving README file](../../README.md) contains general
information on downloading and running the samples.

The Fashion MNIST data set is downloaded automatically when the training script
runs.

## Prerequisites

The inference scripts require `tensorflow-serving-api`, which is not included
in the base IBM Z Accelerated for TensorFlow container. Because the base
container runs as `ibm-user` (non-root), `prerequisites.sh` handles this by
building a new container image on your behalf:

1. It passes your chosen TensorFlow production image as a build argument to
   `Containerfile`.
2. `Containerfile` temporarily switches to `root` to install
   `tensorflow-serving-api` into the venv, then drops back to `ibm-user` as
   the runtime user.
3. Once the image is built, `prerequisites.sh` creates a `workspace/`
   directory alongside the sample scripts, then starts an interactive shell
   inside the container with:
   - The sample scripts mounted read-only at `/scripts`
   - The `workspace/` directory mounted at `/workspace` (writable)

Run the script on the **host** (not from inside a container), passing your
IBM Z Accelerated for TensorFlow production image as the argument:

```bash
./prerequisites.sh <tf-base-image>
```

For example:

```bash
./prerequisites.sh icr.io/ibmz/ibmz-accelerated-for-tensorflow:1.6.0
```

This builds a local image tagged `fashion-mnist-serving-sample:latest` and
drops you into an interactive shell at `/workspace` inside the container. All
output files (trained model, exported SavedModel, etc.) are written there.

## Running the Sample

### Step 1 — Train and export the model

All commands in this step are run from inside the container started by
`prerequisites.sh`, where `/workspace` is the working directory.

Train the model and export it as a SavedModel:

```bash
python /scripts/fashion_mnist_training.py
```

This saves the exported model to `./saved_model/1` inside `/workspace`. Once
training is complete, exit the container:

```bash
exit
```

### Step 2 — Start the TensorFlow Serving container

Run the IBM Z Accelerated for TensorFlow Serving container on the **host**,
mounting the exported model from `workspace/saved_model`:

```bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
docker run -d --rm \
    -p 8500:8500 \
    -p 8501:8501 \
    -v "${SCRIPT_DIR}/workspace/saved_model:/models/fashion_mnist:z" \
    -e MODEL_NAME=fashion_mnist \
    icr.io/ibmz/ibmz-accelerated-serving-for-tensorflow:X.X.X
```

- `--detach` runs the serving container in the background.
- `-p 8500:8500` and `-p 8501:8501` publish the gRPC and REST ports.
- `-v` mounts the exported SavedModel into the serving container.
- `-e MODEL_NAME=fashion_mnist` tells TensorFlow Serving what to call the model.

You can verify the server is ready by querying its metadata:

```bash
curl http://localhost:8501/v1/models/fashion_mnist/metadata
```

### Step 3 — Run inference

Re-enter the sample container for inference, passing `inference` as the second
argument so `prerequisites.sh` adds `--network=host` to the container run:

```bash
./prerequisites.sh <tf-base-image> inference
```

Then run inference over gRPC:

```bash
python /scripts/fashion_mnist_grpc.py
```

Or over REST:

```bash
python /scripts/fashion_mnist_rest.py
```

Both scripts will report prediction accuracy for sample images.

## Known Issues

There are no known open issues with this sample.
