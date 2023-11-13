# Fashion MNIST Sample

The code sample in this directory
[loads](https://www.tensorflow.org/versions/r2.12/api_docs/python/tf/keras/datasets/fashion_mnist/load_data)
the
[Fashion MNIST data set](https://www.tensorflow.org/datasets/catalog/fashion_mnist)
and trains a model. A second script performs inference on the model with the
test data set and displays the results.

The [tensorflow README file](../../README.md) contains general information on
downloading and running the samples.

These samples will download the MNIST data set from the Internet.

# Running the Sample

These instructions assume you have cloned the repository or have otherwise
copied the files in this directory to your host system so you can access the
code.

## Training and Saving the Model

Note that you will run this commands from inside the IBM Z Accelerated for
TensorFlow container. These steps follow the
[Fashion MNIST training sample for the TensorFlow container](https://github.com/IBM/ibmz-accelerated-for-tensorflow/samples/fashion-mnist),
expect here we show how to retrieve the model so it can be used with TensorFlow
Serving.

With podman as the container engine, some additional setup must be done as
shown.

Note `X.X.X` in these samples refers to the current version of the container
image in IBM Container Registry.

```bash
cd samples

# podman-only setup so TensorFlow can create directories and files from
# training within this directory.
chmod o+rwx fashion-mnist

docker run -it --rm -v ./fashion-mnist/:/home/ibm-user/fashion-mnist:z --workdir /home/ibm-user/fashion-mnist icr.io/ibmz/ibmz-accelerated-for-tensorflow:X.X.X bash
```

- This container specified `-v`, which will bind mount the local folder
  `./fashion-mnist` to the container at `/home/ibm-user/fashion-mnist`. This
  will allow the model files to be accessed for the next step.
- `--workdir` sets the current working directory to the bind mount. The sample
  is coded to save the saved model to the current working directory.

First, train and save the model to disk with the `fashion_mnist_training.py`
script. This will download the fashion MNIST data set and create a model in the
current directory.

Training will take some time. The epoch number in the output will indicate
progress.

```bash
python fashion_mnist_training.py
```

Once complete, you can exit the IBM Z Accelerated for TensorFlow container. The
model has been saved in the bind mounted directory.

```bash
exit
```

With podman, the model files will be owned by the sub-uid used by the container.
To change the ownership back to your user id, enter the following commands.

```bash
# podman-only setup. Note that `root` in this context refers to your
# user id and group, not the real root user on the host machine.
podman unshare chown -R root:root ./fashion-mnist/saved_model

# Confirm that your id now owns the files.
ls -la ./train/saved_model
```

## Serving the Saved Model

Once the model has been trained, run the IBM Z Accelerated for TensorFlow
Serving container to serve the model.

```bash
docker run -it --rm --detach -p 8500:8500 -p 8501:8501 -v './fashion-mnist/saved_model:/models/fashion_mnist:z' -e MODEL_NAME=fashion_mnist icr.io/ibmz/ibmz-accelerated-serving-for-tensorflow:X.X.X
```

- This container has been run with `--detach`, which will run the container in
  the background.
- This container has been run with `-p`, which will publish the ports `8500` and
  `8501` to the container.
- This container has been run with `-v`, which will mount the local folder
  `./fashion-mnist/saved_model` to the container at `/models/fashion_mnist`.
- This container has been run with `-e`, which will set the environment variable
  for `MODEL_NAME` with value `fashion_mnist` to the container.

This will serve the saved model, which can be accessed via ports 8500 (gRPC) and
8501 (REST).

You can query the metadata for the model using curl:

```bash
curl http://localhost:8501/v1/models/fashion_mnist/metadata
```

## Running Inference on Served Model

Note that you will run these commands from inside the IBM Z Accelerated for
TensorFlow container.

```bash
docker run -it --rm --network=host -v './fashion-mnist:/home/ibm-user/fashion-mnist:z' --workdir /home/ibm-user/fashion-mnist icr.io/ibmz/ibmz-accelerated-for-tensorflow:X.X.X bash
```

- This container has been run with `--network=host`, which will add host network
  scope to the container. This is for example purposes, in a production
  environment you should not use `--network=host` for security purposes.

Once the model has served, run the `fashion_mnist_grpc.py` script to run
inference against the model using gRPC.

```bash
# This will install this package from the Internet
pip install tensorflow-serving-api
python fashion_mnist_grpc.py
```

The script will report prediction accuracy for some sample images.

To run inference against the model using REST API, run the
`fashion_mnist_rest.py` script.

```bash
python fashion_mnist_rest.py
```

The script will report prediction accuracy for some sample images.

Once complete, you can exit the IBM Z Accelerated for TensorFlow container.

```bash
exit
```
