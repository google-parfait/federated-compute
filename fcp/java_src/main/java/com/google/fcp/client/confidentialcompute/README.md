# Confidential Compute Client

## Overview

The confidential compute client uploads data from a device to a confidential
aggregation running in a trusted execution environment (TEE). It checks in with
the federated compute server and opens an upload session, verifies the TEE's
attestation against the caller's endorsement options, encrypts the data to the
attested key, and uploads the ciphertext. Only the TEE can decrypt it.

## Requirements

The minimum Android SDK version is 24 (Android 7.0).

-   **API 24 and 25:** the library uses `java.time`, so the app must enable core
    library desugaring.
-   **API 24 to 29:** payload encryption uses AES-GCM-SIV, which the platform
    security provider only implements from API 30. On older versions the app
    must install a provider that implements `AES/GCM-SIV/NoPadding` before the
    first call to `encode`. With Google Play services, call
    `ProviderInstaller.installIfNeeded(context)` (or `installIfNeededAsync`)
    once at startup, for example in `Application.onCreate()`. Without such a
    provider `encode` fails with `PayloadEncodingException` and nothing is
    uploaded. On API 29 the platform provider may already implement it,
    depending on the version of the Conscrypt Mainline module on the device; the
    call is still required because that cannot be relied on.

## Usage Workflow

1.  **Build**: Create a `ConfidentialCompute` with its builder. Set the API key
    and an Android `Context`. The other settings have defaults, including the
    server URL, which comes from `setFlags`.
2.  **Fetch**: Call `workloadSetFetcher().fetch(logSourceName)` for a private
    log source. It checks in, requests task assignments, opens an upload session
    per assigned set and downloads the session's encryption config. It returns
    the workload sets the device may contribute to, each carrying the unverified
    encryption config and endorsements and single-use upload tokens.
3.  **Encode**: For each set, select fresh, uncontributed local data with
    `WorkloadSet.selectorSpec()` and call
    `payloadEncoder().encode(workloadSet, endorsementOptions, inputs)`. This
    verifies the encryption config against your endorsement options, builds the
    checkpoint and encrypts it to the verified key. A verification failure fails
    `encode` with `PayloadEncodingException`.
4.  **Upload**: Call `payloadUploader().upload(workloadSet, payload)`. Then mark
    the data as contributed or roll it back, as described below.

Every stage returns a `ListenableFuture` and runs its blocking work on the
builder's executor, one call at a time.

## Your Data Store: Contribute Each Record Once

The data you pass to `encode` must be fresh: records that have never been
contributed to that workload set. The server cannot detect a record uploaded
twice, so this is the caller's job, and getting it wrong breaks the privacy
guarantees of the workloads.

Your data store must use one of two designs for each workload set.

**Hand each record out once.** After a record has been returned for a set, never
return it for that set again, whatever happens to the upload.

**Roll back when nothing was stored.** Mark records as contributed when you
return them for a set, then undo that only when the payload is known not to have
been stored. `PrivateLogger` does this. Roll back when:

-   the upload fails with `UploadAbortedException`, or
-   encoding fails, or you never call `upload`.

In every other case, including any other upload failure, the records count as
contributed and must not be returned for that set again.

## Code Structure

-   `:api` (`com.google.fcp.client.confidentialcompute`): the stage interfaces
    and value types callers use, such as `WorkloadSet` and `EncodedPayload`.
-   `:confidentialcompute`: `ConfidentialCompute` and its builder, which create
    the default stages.
-   `http`: `HttpUrlConnectionEngine`, for callers that supply their own HTTP
    engine.
-   `internal/*`: the default stages (`fetch`, `encode`, `upload`), crypto, HTTP
    transport and sampling without replacement (`swor`). Visible only within
    this package tree.

Callers depend on `:api` and `:confidentialcompute`, plus `http` if they set an
HTTP engine. No target exports another, so each caller lists the targets whose
types it uses.

```
+----------------------------------------------------------------------+
|                         Application / Caller                         |
+----------------------------------------------------------------------+
            |                                          |
            | Builds                                   | Calls stages
            v                                          v
+------------------------------+        +------------------------------+
| :confidentialcompute         |        | :api                         |
| ConfidentialCompute          |        | WorkloadSetFetcher           |
| ConfidentialCompute.Builder  |        | ConfidentialPayloadEncoder   |
+------------------------------+        | ConfidentialPayloadUploader  |
            |                           | WorkloadSet, EncodedPayload  |
            | Creates                   +------------------------------+
            v                                          ^
                                                       | Implements
+----------------------------------------------------------------------+
|                     internal/* (default stages)                      |
|              fetch, encode, upload, crypto, http, swor               |
+----------------------------------------------------------------------+
```

## Replacing a Stage

Pass your own implementation to `setWorkloadSetFetcher`, `setPayloadEncoder` or
`setPayloadUploader`. The builder then creates defaults only for the other
stages, and a setting such as the API key is required only if a default stage
uses it. A replacement encoder verifies `WorkloadSet.encryptionConfig()` and
`WorkloadSet.signedEndorsements()` against the endorsement options passed to
`encode` and encrypts to the verified key; a replacement uploader sends the
payload with `WorkloadSet.uploadTokens()`.

## Sampling Without Replacement

The default fetcher and uploader apply sampling without replacement (SWOR): a
device does not contribute to the same workload set again within the set's SWOR
period. SWOR is on by default and needs the builder's `Context` to store upload
times; `disableSwor()` turns it off. It is needed until every workload in a
workload set uses privacy IDs. See `internal/swor/SworTracker`.

## Diagnostic Events

The default stages report each step of a check-in, task assignment, encode and
upload as a `DiagnosticEvent`. Each event carries a wall-clock timestamp and a
duration measured with a monotonic clock. Events go to the `StabilityMonitor`
passed to `setStabilityMonitor`; the default monitor logs them to logcat at
verbose level. The client does not send events anywhere itself, so the app
decides whether and where to log them.

Most event kinds correspond to a `PhaseLogger` call in the C++ client, so an app
can report similar metrics for both clients. The kinds may change as the client
evolves.

Replaced stages log nothing unless they do so themselves. Querying the inputs is
not a stage, so code that calls the stages directly logs its own `QUERY_*`
events to the same monitor; the private logger connector does this.
