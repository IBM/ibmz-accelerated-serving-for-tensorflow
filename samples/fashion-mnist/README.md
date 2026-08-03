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
3. Once the image is built, `prerequisites.sh` starts an interactive shell
   inside the container with a named volume mounted at `/workspace` (writable).

Run the script on the **host** (not from inside a container), passing your
IBM Z Accelerated for TensorFlow production image as the argument:

```bash
./prerequisites.sh <tf-base-image>
```

For example:

```bash
./prerequisites.sh icr.io/ibmz/ibmz-accelerated-for-tensorflow:1.6.0
```

This builds a local image and prints the generated image tag (e.g.
`fashion-mnist-serving-sample:20250714-143022`) along with `docker run`
commands to start the container. All output files (trained model, exported
SavedModel, etc.) are written to `/workspace` inside the container.

## Copying Scripts into the Container

Once the container is running, open a second terminal on the host and use
`docker cp` to copy the sample scripts into the container:

```bash
# Find the running container ID
docker ps

# Copy the sample scripts
docker cp fashion_mnist_training.py <container-id>:/workspace/
docker cp fashion_mnist_grpc.py <container-id>:/workspace/
docker cp fashion_mnist_rest.py <container-id>:/workspace/
```

Then return to the container shell to run the sample.

## Running the Sample

### Step 1 — Train and export the model

All commands in this step are run from inside the container started by
`prerequisites.sh`, where `/workspace` is the working directory.

Train the model and export it as a SavedModel:

```bash
python fashion_mnist_training.py
```

This saves the exported model to `./saved_model/1` inside `/workspace`. Once
training is complete, exit the container:

```bash
exit
```

### Step 2 — Start the TensorFlow Serving container

Run the IBM Z Accelerated for TensorFlow Serving container on the **host**,
mounting the workspace volume so the serving container can read the exported
model:

```bash
docker run -d --rm \
    -p 8500:8500 \
    -p 8501:8501 \
    -v fashion-mnist-serving-workspace:/workspace \
    -e MODEL_NAME=fashion_mnist \
    --entrypoint tensorflow_model_server \
    icr.io/ibmz/ibmz-accelerated-serving-for-tensorflow:X.X.X \
    --port=8500 --rest_api_port=8501 \
    --model_name=fashion_mnist \
    --model_base_path=/workspace/saved_model
```

- `--detach` runs the serving container in the background.
- `-p 8500:8500` and `-p 8501:8501` publish the gRPC and REST ports.
- `-v` mounts the workspace volume into the serving container.
- `--model_base_path` points TensorFlow Serving at the exported SavedModel.

You can verify the server is ready by querying its metadata:

```bash
curl http://localhost:8501/v1/models/fashion_mnist/metadata
```

### Step 3 — Run inference

Re-enter the sample container for inference using the `docker run` command
printed by `prerequisites.sh` with `--network=host`. The script prints both
the training and inference variants — use the inference one:

```bash
docker run -it --rm \
    --network=host \
    -v fashion-mnist-serving-workspace:/workspace \
    -w /workspace \
    fashion-mnist-serving-sample:<timestamp> \
    bash
```

Then run inference over gRPC:

```bash
python fashion_mnist_grpc.py
```

Or over REST:

```bash
python fashion_mnist_rest.py
```

Both scripts will report prediction accuracy for sample images.

## Cleanup

When you are finished with the sample, remove the image and workspace volume:

```bash
docker rmi fashion-mnist-serving-sample:<timestamp>
docker volume rm fashion-mnist-serving-workspace
```

## Known Issues

There are no known open issues with this sample.
