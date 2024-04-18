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

## Preparing a device for verification record extraction

1.  Select an Android phone to use. You must be able to enable 'Developer
    mode'/`adb` access on this device (see below).

2.  Install the [`adb` tool](https://developer.android.com/tools/adb) on your
    computer.

3.  [Enable](https://developer.android.com/tools/adb#Enabling) `adb` debugging
    on your device.

4.  Connect your computer to the phone using USB.

5.  Run the following command to turn on the logging of
    `AttestationVerificationRecord` entries to your device's logcat buffer, for
    any app that is using this library.

    ```sh
    adb shell setprop log.tag.fcp.infra D
    ```

6.  Run the following command:

    ```sh
    adb logcat 'fcp.attest:D' '*:S' | tee fcp_attestation_records.txt
    ```

    This will write all log messages containing attestation verification records
    to the `fcp_attestation_records.txt` file, as well as to the terminal
    output. Leave this command running as you proceed to the next step.

7.  Leave your phone connected to your computer using the USB connection, make
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
`oak_rust_attestation_verifier.cc`).

NOTE: We plan to provide a utility for parsing these encoded records into plain
binary protos in the future.

Once you have gathered one or more such records, you can follow the instructions
in the
[Confidential Federated Compute](https://github.com/google-parfait/confidential-federated-compute/inspecting_attestion_records/README.md)
repository to inspect the information contained in those records, and to
identify and inspect the TEE-attested binaries that were allowed to process the
data that was uploaded by this device during the session in which the
verification record was generated.
