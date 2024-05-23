# Remote attestation verification

The code in this directory (`oak_rust_attestation_verifier.cc`) performs remote
attestation verification of TEE-hosted ledger applications which form part of
the
[`ConfidentialAggregations`](/fcp/protos/federatedcompute/confidential_aggregations.proto)
protocol, as described in the
[Confidential Federated Computations paper](https://arxiv.org/abs/2404.10764).

Client devices that use this library to participate in the
`ConfidentialAggregations` protocol will verify attestation evidence for the
ledger application hosted in the
[Confidential Federated Compute](https://github.com/google-parfait/confidential-federated-compute)
repository. They will also verify the data access policy that the ledger will
enforce, which will specify one or more allowed data transformation applications
which are built from that same repository.

After a successful verification, these devices will log an
[attestation verification record](/fcp/protos/confidentialcompute/verification_record.proto)
which can then be inspected.

**This functionality is currently only being tested on a small scale population
in the Android Gboard app**. However, for a device to successfully participate
in this population, various conditions have to be met, among which: the beta
version of the Gboard app has to be installed, the OS scheduler has to invoke
the app, a test needs to be ongoing (we may pause or restart tests at times),
and the server has to admit the device by serving a task.

You can attempt to have a device participate in the population and thereafter
extract the attestation record logged by the device by using the following
instructions. However, note that due to the required conditions listed above,
these instructions do not guarantee that your device will actually participate
within a certain timeframe.

## Preparing a device for verification record extraction

1.  Select an Android phone to use. You must be able to enable 'Developer
    mode'/`adb` access on this device (see below).

2.  Install the app that is using this library and from which you want to gather
    attestation verification records. E.g. for the ongoing test in Gboard, you
    must:

    1.  Install the beta version of Gboard on your device by using
        [this link](https://play.google.com/apps/testing/com.google.android.inputmethod.latin),
    2.  Use the Gboard app at least once on your device.
    3.  Set your primary language in Gboard to "English (United States)".

3.  Install the [`adb` tool](https://developer.android.com/tools/adb) on your
    computer.

4.  [Enable](https://developer.android.com/tools/adb#Enabling) `adb` debugging
    on your device.

5.  Connect your computer to the phone using USB.

6.  Run the following command to turn on the logging of
    `AttestationVerificationRecord` entries to your device's logcat buffer, for
    any app that is using this library.

    ```sh
    adb shell setprop log.tag.fcp.attest D
    ```

7.  Run the following command:

    ```sh
    adb logcat 'fcp.attest:D' '*:S' | tee fcp_attestation_records.txt
    ```

    This will write all log messages containing attestation verification records
    to the `fcp_attestation_records.txt` file, as well as to the terminal
    output. Leave this command running as you proceed to the next step.

8.  Leave your phone connected to your computer using the USB connection, make
    sure it is also connected to WiFi, and leave it unattended. You should not
    interact with the phone in any way at this point and leave the phone for at
    least a number of hours (see the note above). This helps create the required
    (but not necessarily sufficient) conditions for your phone to be able to
    participate in the population.

## Parsing extracted verification records

If one of the apps on the device is using this library, and if the device
successfully participates in a task using the `ConfidentialAggregations`
protocol (subject to the required conditions listed above), then the device's
logcat stream will contain an entry that looks as follows:

```
03-28 12:15:12.563  4101  4123 D fcp.attest: This device is contributing data via the confidential aggregation protocol. The attestation verification record follows.
03-28 12:15:12.563  4101  4123 D fcp.attest: <H4sIAAAAAAAAA+M6Ji51WJzrgLhHak5OvqKCQuIoGAWjYBSMglEwCkbBKBgFo2DQAwDdp4nbyQsAAA==>
...
03-28 12:15:12.563  4101  4123 D fcp.attest: <>
```

These records represent a line-wrapped, base64-encoded, and gzip-compressed
`AttestationVerificationRecord` protobuf (the encoding is performed in
`oak_rust_attestation_verifier.cc`). We provide an `extract_attestation_records`
utility for extracting and parsing such records from a log stream. This utility
will write each extracted record to its own file, in the shape of a serialized
`AttestationVerificationRecord` proto.

```shell
$ mkdir extracted_records
$ bazelisk run :extract_attestation_records -- \
  --input=$PWD/fcp_attestation_records.txt \
  --output=$PWD/extracted_records/
Wrote a verification record extracted from lines 2-19 to extracted_records/record_l2_to_l19_digest12345678.pb
Found 1 record(s).
```

Once you have gathered one or more such records, you can follow the instructions
in the
[Confidential Federated Compute](https://github.com/google-parfait/confidential-federated-compute/blob/main/inspecting_attestation_records/README.md)
repository to inspect the information contained in those records, and to
identify and inspect the TEE-attested binaries that were allowed to process the
data that was uploaded by this device during the session in which the
verification record was generated.
