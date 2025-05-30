# Credit Card Fraud Sample

The code sample in this directory uses the
[Credit Card Fraud data set](https://github.com/IBM/TabFormer/tree/main/data/credit_card)
and deploys a saved model for TensorFlow Serving. A second script performs
inference on the model with the test data set and displays the results.

The [tensorflow README file](../../README.md) contains general information on
downloading and running the samples.

These samples will require first downloading the data set from the Internet and
extracting the archive.

# Running the Sample

These instructions assume you have cloned the repository or have otherwise
copied the files in this directory to your host system so you can access the
code.

## Training and Saving the Model

Follow the steps in
[Credit Card Fraud training sample for the TensorFlow container](https://github.com/IBM/ibmz-accelerated-for-tensorflow/samples/credit-card-fraud),
to create, train, and save the model. Once completed, there should be a
`saved_model` folder in your working directory, along with various artifacts
created when saving the model.

## Deploying the Model for TensorFlow Serving

Note that you will run this commands from inside the IBM Z Accelerated for
TensorFlow container.

With podman as the container engine, some additional setup must be done as
shown.

Note `X.X.X` in these samples refers to the current version of the container
image in IBM Container Registry.

```bash
cd samples

# podman-only setup so TensorFlow can create directories and files from
# training within this directory.
chmod o+rwx credit-card-fraud

docker run -it --rm -v ./credit-card-fraud/:/home/ibm-user/credit-card-fraud:z --workdir /home/ibm-user/credit-card-fraud icr.io/ibmz/ibmz-accelerated-for-tensorflow:X.X.X bash
```

- This container specified `-v`, which will bind mount the local folder
  `./credit-card-fraud` to the container at `/home/ibm-user/credit-card-fraud`.
  This will allow the model files to be accessed for the next step.
- `--workdir` sets the current working directory to the bind mount. The sample
  is coded to save the saved model to the current working directory.

First, deploy the model to a Servable with the `credit_card_fraud_deployment.py`
script. This will create a `serving_model` folder in the current directory.

```bash
# This will install this package from the Internet
pip install tensorflow-serving-api
python credit_card_fraud_deployment.py
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
podman unshare chown -R root:root ./credit-card-fraud/serving_model

# Confirm that your id now owns the files.
ls -la ./credit-card-fraud/serving_model
```

## Serving the Saved Model

Once the model has been trained, run the IBM Z Accelerated for TensorFlow
Serving container to serve the model.

```bash
docker run -it --rm --detach -p 8500:8500 -p 8501:8501 -v './credit-card-fraud/serving_model:/models:z' -e MODEL_NAME=lstm icr.io/ibmz/ibmz-accelerated-serving-for-tensorflow:X.X.X
```

- This container has been run with `--detach`, which will run the container in
  the background.
- This container has been run with `-p`, which will publish the ports `8500` and
  `8501` to the container.
- This container has been run with `-v`, which will mount the local folder
  `./credit-card-fraud/serving_model` to the container at `/models`.
- This container has been run with `-e`, which will set the environment variable
  for `MODEL_NAME` with value `lstm` to the container.

This will serve the saved model, which can be accessed via ports 8500 (gRPC) and
8501 (REST).

You can query the metadata for the model using curl:

```bash
curl http://localhost:8501/v1/models/lstm/metadata
```

## Running Inference on Served Model

Note that you will run these commands from inside the IBM Z Accelerated for
TensorFlow container.

```bash
docker run -it --rm --network=host -v './credit-card-fraud:/home/ibm-user/credit-card-fraud:z' --workdir /home/ibm-user/credit-card-fraud icr.io/ibmz/ibmz-accelerated-for-tensorflow:X.X.X bash
```

- This container has been run with `--network=host`, which will add host network
  scope to the container. This is for example purposes, in a production
  environment you should not use `--network=host` for security purposes.

Once the model has served, run the `credit_card_fraud_grpc.py` script to run
inference against the model using gRPC.

```bash
# This will install this package from the Internet
pip install tensorflow-serving-api
python credit_card_fraud_grpc.py
```

The script will report prediction accuracy for some sample transactions.

To run inference against the model using REST API, run the
`credit_card_fraud_rest.py` script.

```bash
python credit_card_fraud_rest.py
```

The script will report prediction accuracy for some sample transactions.

Once complete, you can exit the IBM Z Accelerated for TensorFlow container.

```bash
exit
```
