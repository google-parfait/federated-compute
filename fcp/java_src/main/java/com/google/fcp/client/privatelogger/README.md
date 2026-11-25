# Private Logger

## Overview

The `PrivateLogger` library provides a verifiably private way to log data to an
on-device store and then upload it to a private log source. When connected to a
Federated Compute protocol implementation and Confidential Federated Compute
platform, it ensures that data is processed in a secure and privacy-preserving
manner.

## Usage Workflow

1.  **Initialization**: Obtain an instance of `PrivateLogger` (typically from a
    glue/integration library).
2.  **Logging**: Call `log(MessageLite)` to add data entries to the on-device
    store. This is an asynchronous operation.
3.  **Uploading**: Calling `upload()` triggers an upload to the configured
    private log source and runs a best-effort retention cleanup of expired
    entries. The result is returned via a `ListenableFuture`. If the future
    fails, callers should expect and gracefully handle
    `PrivateLoggerUnavailableException` for cases where the private logger
    implementation cannot be used (e.g. if the implementation was disabled by
    flag or the upload was cancelled).

## Code Structure

The library is divided into two main packages:

-   `com.google.fcp.client.privatelogger`: Contains the public interfaces and
    configuration classes.
-   `com.google.fcp.client.privatelogger.impl`: Contains the internal
    implementation details, such as the Room database and DAOs.

This separation is intentional and serves to control API visibility. The public
`privatelogger` package is intended for general consumption by any application
that needs to log data. The `impl` package, however, contains concrete classes
like `PrivateLoggerImpl` and `PrivateLoggerBuilder` which are intended to be
used by "glue" libraries that bind the generic logger implementation to a
specific platform's federated compute protocol via a `PrivateLoggerConnector`
implementation.

By separating the packages, we can expose the public interfaces broadly while
restricting access to the implementation details. This prevents end-user
applications from directly instantiating `PrivateLoggerImpl` and ensures that
only trusted integration libraries can construct and configure logger instances
with the necessary platform-specific connectors.

The following diagram illustrates the relationship between the components:

```
+----------------------------------------------------------------------+
|                          Application/End-User                        |
|                  (Depends only on the public API)                    |
+----------------------------------------------------------------------+
                                     |
                                     | Uses
                                     v
+----------------------------------------------------------------------+
|              Public API (`...fcp.client.privatelogger`)              |
|              - PrivateLogger (Interface)                             |
|              - PrivateLoggerConfig                                   |
+----------------------------------------------------------------------+
                                     ^
                                     | Implemented by
                                     |
+----------------------------------------------------------------------+
|        Internal Implementation (`...privatelogger.impl`)             |
|        - PrivateLoggerImpl                                           |
|        - PrivateLoggerBuilder                                        |
|        - PrivateLoggerConnector (Interface)                          |
|        - UploadOutcome                                               |
+----------------------------------------------------------------------+
                                     ^
                                     | Used by
                                     |
+----------------------------------------------------------------------+
|                 Glue Library / Platform Integration                  |
| (Constructs Impl, provides a specific PrivateLoggerConnector impl)   |
+----------------------------------------------------------------------+
```

## Timestamp Handling

The `log` method automatically captures a timestamp for each log entry. This is
stored in a `timestamp` field as a string formatted according to
`YYYY-MM-DDTHH:MM:SS.SSS[+-]HH:MM` (ISO 8601 with milliseconds). This string
preserves the device's local time-zone information at the time of logging and is
used for all internal database operations, such as enforcing the retention
policy.

## Data Retention and Cleanup

Data logged to the on-device store is retained according to the retention period
defined in `PrivateLoggerConfig`. Expired entries are automatically deleted from
the database when `PrivateLogger` is initialized, on a best-effort basis after
each `upload()` call, and periodically during logging (every 1,000 logs since
initialization). The periodic log counter is held in memory and resets each time
the logger is instantiated.
