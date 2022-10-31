
When building libraries (such as custom op libraries) agains the TensorFlow pip
package, care must be taken to ensure those libraries build against that
package's headers and with the same compiler and linker flags as that package
was compiled with. These utilities help ensure that's the case.

First, add the following to your `WORKSPACE` file to configure a repository that
provides the C++ headers and libraries provided by the TensorFlow pip package.

```
load("//fcp/tensorflow/system_provided_tf:system_provided_tf.bzl", "system_provided_tf")
system_provided_tf(name = "system_provided_tf")
```

Then simply load `tf_custom_op_library` from
`@system_provided_tf//:system_provided_tf.bzl` instead of
`@org_tensorflow//tensorflow:tensorflow.bzl`.

NOTE: The `gpu_srcs` and `gpu_deps` parameters supported by TensorFlow's version
of `tf_custom_op_library` are not supported by this version.
