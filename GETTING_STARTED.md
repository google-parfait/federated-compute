# Instructions for getting the Federated Compute Platform code up and running on your own machine.

## Download and install apt-provided dependencies

First run `sudo apt update` to update the package index.

### Basic tools

There are some basic tools and packages you will need on your machine:

```
sudo apt install git
```

## Language support

The FCP codebase contains C++ code, which requires Clang and libc++, as well as
Python code:

```
sudo apt install clang-13 lld-13 libc++-13-dev libc++abi-13-dev python3.9 python3-venv
```

### Install Bazelisk

Bazelisk is used to fetch the correct Bazel binaries necessary to build and run
your FCP code.

Please read https://github.com/bazelbuild/bazelisk#installation.

## Set up your Python environment

You should make sure that your Python bin and lib paths are set to `python3.9`
in the case you have multiple version of Python on your machine.

Then you can set up your Python environment.

```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

## Clone the FCP repository and install requirements

```
git clone https://github.com/google/federated-compute.git
cd federated-compute
pip install -r requirements.txt
```

## Set up the environment variables

Next, configure environment variables that will make Bazel use clang and libc++:

```
export CC=clang-13 $ export BAZEL_CXXOPTS=-stdlib=libc++ BAZEL_LINKOPTS=-stdlib=libc++
```

## Build FCP and run the server test!

WARNING: Building FCP can be slow the first time, on the order of hours. This
happens because all of TensorFlow is built as part of the process.

```
bazelisk test //fcp/demo:server_test
```
