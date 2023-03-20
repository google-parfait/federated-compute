# Instructions for getting the Federated Compute Platform code up and running on your own machine.

## Download and install build dependencies

### Basic tools

There are some basic tools and packages you will need on your machine:

*   Git
*   A C++ compiler (e.g., Clang or GCC, but see note about GCC below)
*   Python 3.9 or greater, including the `venv` module

For example, on Debian:

```
sudo apt install -y git gcc python3 python3-dev python3-venv
```

> ⚠️ The project maintainers internally test with Clang only, so support for
> GCC-based builds is provided only on a best-effort basis and may at times be
> broken.
>
> If using GCC then we recommend using a recent version (e.g., at least as
> recent as what Debian stable uses, preferably newer than that).
>
> If using Clang then please see [Building with Clang](#building-with-clang) for
> further Clang-specific instructions.

### Install Bazelisk

Bazelisk is used to fetch the correct Bazel binaries necessary to build and run
Federated Compute code.

Please read https://github.com/bazelbuild/bazelisk#installation.

## Set up your Python environment

Setting up a virtual Python environment will ensure that Python dependencies
don't conflict or overwrite your existing Python installation. If you have
multiple installed versions of Python, replace `python3` in the following
instructions with the desired version (e.g., `python3.X`).

```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

Note: To exit the virtual environment, run `deactivate`.

## Clone the Federated Compute repository and install Python requirements

```
git clone https://github.com/google/federated-compute.git
cd federated-compute
pip install -r requirements.txt
```

## Build and run the federated program test!

> ⚠️ Many Federated Compute targets depend on TensorFlow, which can take several
> hours to build for the first time. Consider running builds in `screen` or
> `tmux` if you're worried about your terminal closing during this time.
>
> While not required, Bazel's
> [remote build execution](https://bazel.build/remote/rbe) and
> [remote caching](https://bazel.build/remote/caching) features can speed up
> builds.

```
bazelisk test //fcp/demo:federated_program_test
```

### Building with Clang

Use `--config=clang` to build with clang and libc++. On Debian, this requires
installing several additional packages:

```
sudo apt install -y clang lld libc++-dev libc++abi-dev`
```
