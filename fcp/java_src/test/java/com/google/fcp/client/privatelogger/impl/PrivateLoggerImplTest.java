// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.fcp.client.privatelogger.impl;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteFullException;
import androidx.room.Room;
import androidx.test.core.app.ApplicationProvider;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.fcp.client.FakeTimeSource;
import com.google.fcp.client.TimeSource;
import com.google.fcp.client.privatelogger.LogEntry;
import com.google.fcp.client.privatelogger.PrivateLogger;
import com.google.fcp.client.privatelogger.PrivateLoggerConfig;
import com.google.fcp.client.privatelogger.impl.test.ActivityProto;
import com.google.fcp.client.stabilitymonitor.StabilityMonitor;
import com.google.intelligence.fcp.client.FederatedComputation;
import com.google.intelligence.fcp.client.QueryTimeComputationProperties;
import com.google.intelligence.fcp.client.SelectorContext;
import com.google.protobuf.ExtensionRegistryLite;
import com.google.protobuf.MessageLite;
import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TimeZone;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Captor;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;
import org.robolectric.RobolectricTestRunner;

@RunWith(RobolectricTestRunner.class)
public class PrivateLoggerImplTest {
  @Rule public final MockitoRule mocks = MockitoJUnit.rule();

  private Context context;
  private PrivateLogger logger;
  private final Executor executor = directExecutor();
  private final FakeTimeSource fakeTimeSource = new FakeTimeSource();
  private TimeZone originalDefaultTimeZone;

  @Mock private PrivateLoggerConnector mockConnector;
  @Mock private StabilityMonitor mockStabilityMonitor;
  @Captor private ArgumentCaptor<DataProvider> dataProviderCaptor;

  private static final String TEST_PRIVATE_LOG_SOURCE = "test/private_log_source";
  private static final Duration TEST_RETENTION = Duration.ofDays(1);
  private static final String TASK_NAME = "testTask";

  @Before
  public void setUp() {
    originalDefaultTimeZone = TimeZone.getDefault();
    context = ApplicationProvider.getApplicationContext();
    logger = createLogger();
  }

  @After
  public void tearDown() throws Exception {
    TimeZone.setDefault(originalDefaultTimeZone);
    File dbFile = context.getDatabasePath("task-contribution-database");
    if (dbFile.exists()) {
      dbFile.delete();
    }
    context.deleteDatabase(PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE));
    context.deleteDatabase(PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE + "_fail"));
  }

  @Test
  public void testUpload() throws Exception {
    SettableFuture<List<UploadOutcome>> uploadFuture = SettableFuture.create();
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), any(DataProvider.class)))
        .thenReturn(uploadFuture);

    ListenableFuture<Void> future = logger.upload();

    // Simulate immediate successful upload completion.
    assertThat(future.isDone()).isFalse();
    uploadFuture.set(
        ImmutableList.of(
            UploadOutcome.builder()
                .setTaskName(TASK_NAME)
                .setStatus(UploadOutcome.Status.CONTRIBUTED)
                .build()));
    assertThat(future.isDone()).isTrue();
    assertThat(future.get()).isNull();
  }

  @Test
  public void testLog_insertsEntry() throws Exception {
    ActivityProto testProto = ActivityProto.getDefaultInstance();
    // Set a timezone to ensure we're correctly formatting the timestamp string from epoch millis.
    TimeZone.setDefault(TimeZone.getTimeZone("America/New_York"));
    Instant logTime = fakeTimeSource.now();

    logger.log(testProto);

    // Trigger an upload and verify that the connector is invoked with a data provider.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused = logger.upload();

    // Get data from the provider and verify its contents.
    DataProvider dataProvider = dataProviderCaptor.getValue();
    List<LogEntry> entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get(); // Blocks until the future is done.

    // Verify that the retrieved entry matches what we logged.
    assertThat(entries).hasSize(1);
    LogEntry retrievedEntry = entries.get(0);
    assertThat(retrievedEntry.value().toByteArray()).isEqualTo(testProto.toByteArray());
    String expectedTimestamp = TimestampUtil.getIso8601String(logTime);
    assertThat(retrievedEntry.timestamp()).isEqualTo(expectedTimestamp);
  }

  @Test
  public void testLog_exceptionReportedToStabilityMonitor() throws Exception {
    logger.log((MessageLite) null);

    verify(mockStabilityMonitor).reportCrash(any(NullPointerException.class));
  }

  @Test
  public void testLog_sequentialExecutorRecoversAfterException() throws Exception {
    // Calling log(null) should cause an exception, which will be caught and logged.
    logger.log((MessageLite) null);

    // Now log a valid entry. If the sequential executor halts after the first exception,
    // then this entry will not be added to the DB.
    ActivityProto testProto = ActivityProto.getDefaultInstance();
    logger.log(testProto);

    // Trigger an upload and verify that the valid entry is present.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused = logger.upload();

    DataProvider dataProvider = dataProviderCaptor.getValue();
    List<LogEntry> entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();

    // Verify that the retrieved entry matches the valid proto we logged.
    assertThat(entries).hasSize(1);
    LogEntry retrievedEntry = entries.get(0);
    assertThat(retrievedEntry.value().toByteArray()).isEqualTo(testProto.toByteArray());
  }

  @Test
  public void testCreate_cleansUpOldEntries() throws Exception {
    // Log an entry that will become "old".
    logger.log(ActivityProto.getDefaultInstance());

    // Advance time past the retention period.
    fakeTimeSource.advance(TEST_RETENTION.plusDays(1));

    // Now, create a new logger instance, which should trigger cleanup of the old entry. Then log a
    // new entry, which should be the only one remaining.
    logger = createLogger();
    ActivityProto recentProto = ActivityProto.getDefaultInstance();
    Instant logTime = fakeTimeSource.now();
    logger.log(recentProto);

    // Trigger an upload and verify via the data provider that only the recent entry is available.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused = logger.upload();
    DataProvider dataProvider = dataProviderCaptor.getValue();
    List<LogEntry> entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();

    // Verify that the retrieved entry is the recent one.
    assertThat(entries).hasSize(1);
    LogEntry retrievedEntry = entries.get(0);
    assertThat(retrievedEntry.value().toByteArray()).isEqualTo(recentProto.toByteArray());
    String expectedTimestamp = TimestampUtil.getIso8601String(logTime);
    assertThat(retrievedEntry.timestamp()).isEqualTo(expectedTimestamp);
  }

  @Test
  public void testUpload_filtersOutEntriesThatExpireAfterCreation() throws Exception {
    // Log an entry that will expire after logger creation but before upload.
    ActivityProto oldProto = ActivityProto.newBuilder().setDurationSeconds(123).build();
    logger.log(oldProto);

    // Advance time, but not past the retention period.
    fakeTimeSource.advance(TEST_RETENTION.dividedBy(2));

    // Log a more recent entry that should not expire.
    ActivityProto recentProto = ActivityProto.newBuilder().setDurationSeconds(456).build();
    Instant logTime = fakeTimeSource.now();
    logger.log(recentProto);

    // Advance time past the retention period *for the first entry*. The second entry is still
    // within the retention period.
    fakeTimeSource.advance(TEST_RETENTION.dividedBy(2).plusSeconds(1));

    // Now, trigger an upload. The data provider created during this upload should filter out the
    // first (now-expired) entry.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused = logger.upload();
    DataProvider dataProvider = dataProviderCaptor.getValue();
    List<LogEntry> entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();

    // Verify that the retrieved entry is the recent one.
    assertThat(entries).hasSize(1);
    LogEntry retrievedEntry = entries.get(0);
    assertThat(retrievedEntry.value().toByteArray()).isEqualTo(recentProto.toByteArray());
    String expectedTimestamp = TimestampUtil.getIso8601String(logTime);
    assertThat(retrievedEntry.timestamp()).isEqualTo(expectedTimestamp);
  }

  @Test
  public void testUpload_cleansUpOldEntriesPostUpload() throws Exception {
    // 1. Log an entry that will become expired.
    ActivityProto oldProto = ActivityProto.newBuilder().setDurationSeconds(123).build();
    logger.log(oldProto);

    // 2. Advance time past the retention period.
    fakeTimeSource.advance(TEST_RETENTION.plusDays(1));

    // 3. Trigger upload. Post-upload cleanup should delete the expired entry.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), any(DataProvider.class)))
        .thenReturn(immediateFuture(ImmutableList.of()));
    logger.upload().get();

    // 4. Verify the expired entry was pruned by post-upload cleanup.
    PrivateLoggerDatabase db =
        Room.databaseBuilder(
                context,
                PrivateLoggerDatabase.class,
                PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE))
            .allowMainThreadQueries()
            .build();
    assertThat(db.logEntryDao().getAllAfter(0)).isEmpty();
    db.close();
  }

  @Test
  public void testUpload_cleansUpOldEntriesPostUpload_whenUploadFails() throws Exception {
    // 1. Log an entry that will become expired.
    ActivityProto oldProto = ActivityProto.newBuilder().setDurationSeconds(123).build();
    logger.log(oldProto);

    // 2. Advance time past the retention period.
    fakeTimeSource.advance(TEST_RETENTION.plusDays(1));

    // 3. Mock upload to fail.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), any(DataProvider.class)))
        .thenReturn(immediateFailedFuture(new IOException("Upload failed")));

    // 4. Verify upload() propagates the exception.
    ListenableFuture<Void> uploadFuture = logger.upload();
    ExecutionException exception = assertThrows(ExecutionException.class, uploadFuture::get);
    assertThat(exception).hasCauseThat().isInstanceOf(IOException.class);

    // 5. Verify expired entry was still cleaned up despite upload failure.
    PrivateLoggerDatabase db =
        Room.databaseBuilder(
                context,
                PrivateLoggerDatabase.class,
                PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE))
            .allowMainThreadQueries()
            .build();
    assertThat(db.logEntryDao().getAllAfter(0)).isEmpty();
    db.close();
  }

  @Test
  public void testUpload_succeedsEvenIfCleanupFails() throws Exception {
    // Create a mock TimeSource that throws during the cleanup triggered by upload().
    TimeSource mockTimeSource = mock(TimeSource.class);
    when(mockTimeSource.now())
        .thenReturn(Instant.EPOCH) // create() initial cleanup
        .thenReturn(Instant.EPOCH) // log() timestamp
        .thenThrow(new RuntimeException("cleanup failure")); // upload() cleanup

    PrivateLoggerConfig config =
        PrivateLoggerConfig.builder()
            .setPrivateLogSourceName(TEST_PRIVATE_LOG_SOURCE + "_fail")
            .setRetentionPeriod(TEST_RETENTION)
            .build();
    PrivateLogger failLogger =
        PrivateLoggerImpl.create(
            PrivateLoggerImplDeps.newBuilder()
                .setContext(context)
                .setConfig(config)
                .setConnector(mockConnector)
                .setDbExecutor(executor)
                .setTimeSource(mockTimeSource)
                .setStabilityMonitor(mockStabilityMonitor)
                .build());

    failLogger.log(ActivityProto.getDefaultInstance());

    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE + "_fail"), any(DataProvider.class)))
        .thenReturn(immediateFuture(ImmutableList.of()));

    ListenableFuture<Void> uploadFuture = failLogger.upload();
    assertThat(uploadFuture.get()).isNull();

    // Verify the cleanup failure was reported to the stability monitor.
    verify(mockStabilityMonitor).reportCrash(any(RuntimeException.class));
  }

  @Test
  public void testUpload_databaseErrorInCleanup_isNotReportedAsCrash() throws Exception {
    TimeSource mockTimeSource = mock(TimeSource.class);
    when(mockTimeSource.now())
        .thenReturn(Instant.EPOCH) // create() initial cleanup
        .thenReturn(Instant.EPOCH) // log() timestamp
        .thenThrow(new SQLiteFullException("disk full")); // upload() cleanup
    PrivateLoggerConfig config =
        PrivateLoggerConfig.builder()
            .setPrivateLogSourceName(TEST_PRIVATE_LOG_SOURCE + "_full")
            .setRetentionPeriod(TEST_RETENTION)
            .build();
    PrivateLogger fullLogger =
        PrivateLoggerImpl.create(
            PrivateLoggerImplDeps.newBuilder()
                .setContext(context)
                .setConfig(config)
                .setConnector(mockConnector)
                .setDbExecutor(executor)
                .setTimeSource(mockTimeSource)
                .setStabilityMonitor(mockStabilityMonitor)
                .build());
    fullLogger.log(ActivityProto.getDefaultInstance());
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE + "_full"), any(DataProvider.class)))
        .thenReturn(immediateFuture(ImmutableList.of()));

    assertThat(fullLogger.upload().get()).isNull();

    verify(mockStabilityMonitor, never()).reportCrash(any());
  }

  @Test
  public void testUpload_databaseErrorInRead_failsReadWithIoExceptionAndIsNotReported()
      throws Exception {
    TimeSource mockTimeSource = mock(TimeSource.class);
    when(mockTimeSource.now())
        .thenReturn(Instant.EPOCH) // create() initial cleanup
        .thenReturn(Instant.EPOCH) // log() timestamp
        .thenThrow(new SQLiteFullException("disk full")); // the read, then upload() cleanup
    PrivateLoggerConfig config =
        PrivateLoggerConfig.builder()
            .setPrivateLogSourceName(TEST_PRIVATE_LOG_SOURCE + "_read")
            .setRetentionPeriod(TEST_RETENTION)
            .build();
    PrivateLogger readLogger =
        PrivateLoggerImpl.create(
            PrivateLoggerImplDeps.newBuilder()
                .setContext(context)
                .setConfig(config)
                .setConnector(mockConnector)
                .setDbExecutor(executor)
                .setTimeSource(mockTimeSource)
                .setStabilityMonitor(mockStabilityMonitor)
                .build());
    readLogger.log(ActivityProto.getDefaultInstance());
    SelectorContext selectorContext =
        SelectorContext.newBuilder()
            .setComputationProperties(
                QueryTimeComputationProperties.newBuilder()
                    .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
            .build();
    // A connector that fails with whatever the read fails with.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE + "_read"), any(DataProvider.class)))
        .thenAnswer(invocation ->
            Futures.transform(
                ((DataProvider) invocation.getArgument(1)).getData(selectorContext),
                entries -> ImmutableList.<UploadOutcome>of(),
                directExecutor()));

    ListenableFuture<Void> uploadFuture = readLogger.upload();

    ExecutionException e = assertThrows(ExecutionException.class, uploadFuture::get);
    assertThat(e).hasCauseThat().isInstanceOf(IOException.class);
    assertThat(e).hasCauseThat().hasCauseThat().isInstanceOf(SQLiteFullException.class);
    verify(mockStabilityMonitor, never()).reportCrash(any());
  }

  @Test
  public void testLog_cleansUpOldEntriesPeriodicallyEvery1000Logs() throws Exception {
    // 1. Log an entry that will become expired.
    logger.log(ActivityProto.getDefaultInstance());

    // 2. Advance time past the retention period.
    fakeTimeSource.advance(TEST_RETENTION.plusDays(1));

    // 3. Log 998 more entries (999 total logs so far).
    for (int i = 0; i < PrivateLoggerImpl.LOGS_PER_CLEANUP - 2; i++) {
      logger.log(ActivityProto.getDefaultInstance());
    }

    // Verify expired entry is still present before the 1,000th log.
    PrivateLoggerDatabase db =
        Room.databaseBuilder(
                context,
                PrivateLoggerDatabase.class,
                PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE))
            .allowMainThreadQueries()
            .build();
    assertThat(db.logEntryDao().getAllAfter(0)).hasSize(PrivateLoggerImpl.LOGS_PER_CLEANUP - 1);

    // 4. Log the 1,000th entry, which triggers periodic cleanup.
    logger.log(ActivityProto.getDefaultInstance());

    // 5. Verify the expired entry was pruned, leaving only the 999 recent entries.
    List<DbLogEntry> remainingEntries = db.logEntryDao().getAllAfter(0);
    assertThat(remainingEntries).hasSize(PrivateLoggerImpl.LOGS_PER_CLEANUP - 1);
    assertThat(remainingEntries.get(0).id).isNotEqualTo(1);
    // 6. Verify counter reset: after triggering cleanup at the 1,000th log, logCount resets to 0.
    assertThat(((PrivateLoggerImpl) logger).getLogCount()).isEqualTo(0);

    // 7. Verify subsequent logging continues incrementing from 0.
    logger.log(ActivityProto.getDefaultInstance());
    assertThat(((PrivateLoggerImpl) logger).getLogCount()).isEqualTo(1);
    db.close();
  }

  private PrivateLogger createLogger() {
    PrivateLoggerConfig config =
        PrivateLoggerConfig.builder()
            .setPrivateLogSourceName(TEST_PRIVATE_LOG_SOURCE)
            .setRetentionPeriod(TEST_RETENTION)
            .build();
    return PrivateLoggerImpl.create(
        PrivateLoggerImplDeps.newBuilder()
            .setContext(context)
            .setConfig(config)
            .setConnector(mockConnector)
            .setDbExecutor(executor)
            .setTimeSource(fakeTimeSource)
            .setStabilityMonitor(mockStabilityMonitor)
            .build());
  }

  @Test
  public void testLog_multipleEntriesAreContributionTracked() throws Exception {
    // Log 3 entries.
    ActivityProto testProto1 = ActivityProto.getDefaultInstance();
    logger.log(testProto1);
    ActivityProto testProto2 = ActivityProto.getDefaultInstance();
    logger.log(testProto2);
    ActivityProto testProto3 = ActivityProto.getDefaultInstance();
    logger.log(testProto3);

    // Trigger an upload and verify that the connector is invoked with a data provider.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused1 = logger.upload();

    // Get data from the provider and verify its contents.
    DataProvider dataProvider = dataProviderCaptor.getValue();
    List<LogEntry> entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();

    // Verify that the retrieved entries match what we logged.
    assertThat(entries).hasSize(3);
    assertThat(entries.get(0).value().toByteArray()).isEqualTo(testProto1.toByteArray());
    assertThat(entries.get(1).value().toByteArray()).isEqualTo(testProto2.toByteArray());
    assertThat(entries.get(2).value().toByteArray()).isEqualTo(testProto3.toByteArray());

    // Now trigger another upload. This time there should be no entries, since they were all
    // contribution-tracked.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused2 = logger.upload();
    DataProvider dataProvider2 = dataProviderCaptor.getValue();
    List<LogEntry> entries2 =
        dataProvider2
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();
    assertThat(entries2).isEmpty();
  }

  @Test
  public void testUpload_canBeCalledAgainAfterCompletion() throws Exception {
    // Start the first upload and immediately complete it.
    SettableFuture<List<UploadOutcome>> uploadFuture1 = SettableFuture.create();
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(uploadFuture1);
    ListenableFuture<Void> future1 = logger.upload();
    uploadFuture1.set(ImmutableList.of());
    future1.get(); // Wait for completion.

    // Now log a new entry.
    ActivityProto testProto = ActivityProto.getDefaultInstance();
    logger.log(testProto);

    // Start a second upload. This should trigger a new call to the connector.
    SettableFuture<List<UploadOutcome>> uploadFuture2 = SettableFuture.create();
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(uploadFuture2);
    ListenableFuture<Void> future2 = logger.upload();
    uploadFuture2.set(ImmutableList.of());
    future2.get();

    // Verify the connector was called twice, and the second call's data provider contains the new
    // entry.
    verify(mockConnector, times(2)).upload(eq(TEST_PRIVATE_LOG_SOURCE), any(DataProvider.class));
    List<LogEntry> entries =
        dataProviderCaptor
            .getValue()
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();
    assertThat(entries).hasSize(1);
    assertThat(entries.get(0).value().toByteArray()).isEqualTo(testProto.toByteArray());
  }

  @Test
  public void testLog_whenTimezoneChanges_entriesAreSortedByAbsoluteTime() throws Exception {
    // Ensure we exercise timezone changes that would produce out-of-order timestamps if sorted
    // lexicographically.
    // Paris is UTC+1 in winter, New York is UTC-5 in winter.
    // If we log t1 in Paris, and t2=t1+1sec in New York, t1's timestamp string will be like
    // 14:46:40+01:00, while t2's will be 08:46:41-05:00. Lexicographically, t2 < t1,
    // even though t1 occurred before t2.
    TimeZone.setDefault(TimeZone.getTimeZone("Europe/Paris"));
    logger.log(ActivityProto.newBuilder().setDurationSeconds(100).build());
    fakeTimeSource.advance(Duration.ofSeconds(1));

    TimeZone.setDefault(TimeZone.getTimeZone("America/New_York"));
    logger.log(ActivityProto.newBuilder().setDurationSeconds(200).build());
    fakeTimeSource.advance(Duration.ofSeconds(1));

    TimeZone.setDefault(TimeZone.getTimeZone("Europe/Paris"));
    logger.log(ActivityProto.newBuilder().setDurationSeconds(300).build());

    // Trigger an upload and verify that the connector is invoked with a data provider.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused = logger.upload();

    // Get data from the provider and verify its contents.
    DataProvider dataProvider = dataProviderCaptor.getValue();
    List<LogEntry> entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();

    // Verify that the retrieved entries are sorted according to absolute time, not
    // lexicographically by timestamp string.
    assertThat(entries).hasSize(3);
    assertThat(
            ActivityProto.parseFrom(
                    entries.get(0).value(), ExtensionRegistryLite.getEmptyRegistry())
                .getDurationSeconds())
        .isEqualTo(100);
    assertThat(
            ActivityProto.parseFrom(
                    entries.get(1).value(), ExtensionRegistryLite.getEmptyRegistry())
                .getDurationSeconds())
        .isEqualTo(200);
    assertThat(
            ActivityProto.parseFrom(
                    entries.get(2).value(), ExtensionRegistryLite.getEmptyRegistry())
                .getDurationSeconds())
        .isEqualTo(300);
  }

  @Test
  public void testContributionTracking_cleanupOccursAcrossTimezones() throws Exception {
    // This test verifies that LogEntry and EntryContribution cleanup works correctly even
    // when the timezone changes between logging and cleanup. It uses epoch millis for comparisons,
    // so it should be unaffected by timezone.

    // Set retention to 1 second for this test.
    Duration oneSecondRetention = Duration.ofSeconds(1);
    PrivateLoggerConfig config =
        PrivateLoggerConfig.builder()
            .setPrivateLogSourceName(TEST_PRIVATE_LOG_SOURCE)
            .setRetentionPeriod(oneSecondRetention)
            .build();
    logger =
        PrivateLoggerImpl.create(
            PrivateLoggerImplDeps.newBuilder()
                .setContext(context)
                .setConfig(config)
                .setConnector(mockConnector)
                .setDbExecutor(executor)
                .setTimeSource(fakeTimeSource)
                .setStabilityMonitor(mockStabilityMonitor)
                .build());

    // 1. Log and contribute entry1 in Paris timezone.
    // The timestamp string will be lexicographically large but chronologically early.
    // 1970-01-12T13:46:40Z is used by default FakeTimeSource, which is 14:46:40+0100 in Paris
    // during
    // winter.
    TimeZone.setDefault(TimeZone.getTimeZone("Europe/Paris"));
    ActivityProto testProto1 = ActivityProto.newBuilder().setDurationSeconds(100).build();
    logger.log(testProto1);
    // Ensure entry1 is contributed.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    logger.upload().get();
    // Clear captured value before next upload().
    dataProviderCaptor
        .getValue()
        .getData(
            SelectorContext.newBuilder()
                .setComputationProperties(
                    QueryTimeComputationProperties.newBuilder()
                        .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                .build())
        .get();

    // 2. Advance time past retention and switch to NY timezone.
    fakeTimeSource.advance(oneSecondRetention.plusMillis(100));
    TimeZone.setDefault(TimeZone.getTimeZone("America/New_York"));

    // 3. Create a new logger. This will trigger cleanup.
    // Both EntryContributionDao.deleteOlderThan and LogEntryDao.deleteOlderThan should delete
    // entry1 because cleanup is based on epoch millis comparison.
    logger =
        PrivateLoggerImpl.create(
            PrivateLoggerImplDeps.newBuilder()
                .setContext(context)
                .setConfig(config)
                .setConnector(mockConnector)
                .setDbExecutor(executor)
                .setTimeSource(fakeTimeSource)
                .setStabilityMonitor(mockStabilityMonitor)
                .build());

    // 4. Trigger another upload. Because entry1's DbLogEntry was deleted by
    // LogEntryDao.deleteOlderThan during logger creation in step 3, it should not be available
    // for upload.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    logger.upload().get();

    // Verify entry1 is NOT contributed again, because it was cleaned up.
    List<LogEntry> entries2 =
        dataProviderCaptor
            .getValue()
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();
    assertThat(entries2).isEmpty();
  }

  @Test
  public void testUpload_clearsInMemoryContributionTrackingAfterCompletion() throws Exception {
    // Log an entry.
    logger.log(ActivityProto.getDefaultInstance());

    // First upload: contribute the entry.
    SettableFuture<List<UploadOutcome>> uploadFuture1 = SettableFuture.create();
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(uploadFuture1);
    ListenableFuture<Void> future1 = logger.upload();

    // Invoke the data provider to commit contributions.
    DataProvider dataProvider1 = dataProviderCaptor.getValue();
    List<LogEntry> entries1 =
        dataProvider1
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();
    assertThat(entries1).hasSize(1);

    // Complete the upload with NOT_CONTRIBUTED so rollback is attempted.
    // Since clearCommittedContributions runs after rollbacks, the rollback should succeed, but
    // subsequent rollback attempts should be no-ops.
    uploadFuture1.set(
        ImmutableList.of(
            UploadOutcome.builder()
                .setTaskName(TASK_NAME)
                .setStatus(UploadOutcome.Status.NOT_CONTRIBUTED)
                .build()));
    future1.get();

    // The entry should have been rolled back from the DB, so it's available for re-upload.
    // Start a second upload and verify the entry is available again.
    SettableFuture<List<UploadOutcome>> uploadFuture2 = SettableFuture.create();
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(uploadFuture2);
    ListenableFuture<Void> future2 = logger.upload();

    DataProvider dataProvider2 = dataProviderCaptor.getValue();
    List<LogEntry> entries2 =
        dataProvider2
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();

    // The entry should be available again since it was rolled back.
    assertThat(entries2).hasSize(1);

    // Complete the second upload successfully.
    uploadFuture2.set(
        ImmutableList.of(
            UploadOutcome.builder()
                .setTaskName(TASK_NAME)
                .setStatus(UploadOutcome.Status.CONTRIBUTED)
                .build()));
    future2.get();

    // After successful upload, the in-memory map should be cleared. Verify by starting a third
    // upload — the entry should NOT be available since it was successfully contributed and the
    // contribution is in the DB.
    SettableFuture<List<UploadOutcome>> uploadFuture3 = SettableFuture.create();
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(uploadFuture3);
    ListenableFuture<Void> future3 = logger.upload();

    DataProvider dataProvider3 = dataProviderCaptor.getValue();
    List<LogEntry> entries3 =
        dataProvider3
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();

    // Entry was successfully contributed in upload 2, so it should be filtered out.
    assertThat(entries3).isEmpty();

    uploadFuture3.set(ImmutableList.of());
    future3.get();
  }

  @Test
  public void testUpload_cancelReturnedFuture_doesNotCancelUploadAndRollsBackOnNotContributed()
      throws Exception {
    logger.log(ActivityProto.getDefaultInstance());

    // Start an upload with a connector whose future is not yet complete.
    SettableFuture<List<UploadOutcome>> connectorFuture = SettableFuture.create();
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(connectorFuture);
    ListenableFuture<Void> uploadFuture = logger.upload();

    // Fetch entries so the entry is committed to the session for this task.
    List<LogEntry> entries =
        dataProviderCaptor
            .getValue()
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();
    assertThat(entries).hasSize(1);

    // Cancel the future upload() returned.
    assertThat(uploadFuture.cancel(/* mayInterruptIfRunning= */ true)).isTrue();
    assertThat(uploadFuture.isCancelled()).isTrue();

    // The connector future must not be cancelled, and completing it with NOT_CONTRIBUTED rolls
    // back the contributions.
    assertThat(connectorFuture.isCancelled()).isFalse();
    connectorFuture.set(
        ImmutableList.of(
            UploadOutcome.builder()
                .setTaskName(TASK_NAME)
                .setStatus(UploadOutcome.Status.NOT_CONTRIBUTED)
                .build()));
    assertThat(connectorFuture.isCancelled()).isFalse();

    // Verify entries were rolled back: the next upload should return them again.
    SettableFuture<List<UploadOutcome>> connectorFuture2 = SettableFuture.create();
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(connectorFuture2);
    ListenableFuture<Void> uploadFuture2 = logger.upload();

    List<LogEntry> entries2 =
        dataProviderCaptor
            .getValue()
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();
    assertThat(entries2).hasSize(1);

    connectorFuture2.set(ImmutableList.of());
    uploadFuture2.get();
  }

  @Test
  public void testMigration_fromSharedTracker_importsContributions() throws Exception {
    // 1. Create a legacy centralized task contribution tracker database with existing
    // contributions before creating logger.
    createLegacyCentralizedTaskContributionTrackerDatabase(
        TEST_PRIVATE_LOG_SOURCE, TASK_NAME, ImmutableList.of(1L));

    context.deleteDatabase(PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE));
    logger = createLogger();

    // 2. Log 2 entries into the new logger. Entry 1 will have id=1 (matching the migrated
    // contribution), and Entry 2 will have id=2.
    ActivityProto proto1 = ActivityProto.newBuilder().setDurationSeconds(100).build();
    ActivityProto proto2 = ActivityProto.newBuilder().setDurationSeconds(200).build();
    logger.log(proto1);
    logger.log(proto2);

    // 3. Retrieve data for the task. Entry 1 should be filtered out as already contributed,
    // only Entry 2 should be returned.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused = logger.upload();

    DataProvider dataProvider = dataProviderCaptor.getValue();
    List<LogEntry> entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();

    assertThat(entries).hasSize(1);
    assertThat(
            ActivityProto.parseFrom(
                    entries.get(0).value(), ExtensionRegistryLite.getEmptyRegistry())
                .getDurationSeconds())
        .isEqualTo(200);
  }

  @Test
  public void testMigration_fromSharedTracker_ignoresOtherLogSources() throws Exception {
    // 1. Create a legacy centralized task contribution tracker database with contributions for a
    // DIFFERENT log source.
    createLegacyCentralizedTaskContributionTrackerDatabase(
        "other/log_source", TASK_NAME, ImmutableList.of(1L));

    context.deleteDatabase(PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE));
    logger = createLogger();

    // 2. Log an entry with id=1 into our logger.
    ActivityProto proto1 = ActivityProto.newBuilder().setDurationSeconds(100).build();
    logger.log(proto1);

    // 3. Retrieve data. The entry should NOT be filtered out because the contribution belonged
    // to a different log source.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused = logger.upload();

    DataProvider dataProvider = dataProviderCaptor.getValue();
    List<LogEntry> entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();

    assertThat(entries).hasSize(1);
  }

  @Test
  public void testMigration_idempotent() throws Exception {
    // 1. Create legacy centralized task contribution tracker DB with contributions.
    createLegacyCentralizedTaskContributionTrackerDatabase(
        TEST_PRIVATE_LOG_SOURCE, TASK_NAME, ImmutableList.of(1L));

    context.deleteDatabase(PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE));
    logger = createLogger();

    // 2. Run migration again manually (should be a no-op since it was marked completed).
    ((PrivateLoggerImpl) logger).migrateFromSharedTracker().get();

    // 3. Log entries and verify behavior is consistent without error.
    ActivityProto proto1 = ActivityProto.newBuilder().setDurationSeconds(100).build();
    ActivityProto proto2 = ActivityProto.newBuilder().setDurationSeconds(200).build();
    logger.log(proto1);
    logger.log(proto2);

    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused = logger.upload();

    DataProvider dataProvider = dataProviderCaptor.getValue();
    List<LogEntry> entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();

    assertThat(entries).hasSize(1);
    assertThat(
            ActivityProto.parseFrom(
                    entries.get(0).value(), ExtensionRegistryLite.getEmptyRegistry())
                .getDurationSeconds())
        .isEqualTo(200);
  }

  @Test
  public void testUpload_multipleTasksTrackContributionsIndependently() throws Exception {
    String task1 = "task_alpha";
    String task2 = "task_beta";

    ActivityProto proto1 = ActivityProto.newBuilder().setDurationSeconds(100).build();
    ActivityProto proto2 = ActivityProto.newBuilder().setDurationSeconds(200).build();
    logger.log(proto1);
    logger.log(proto2);

    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused = logger.upload();
    DataProvider dataProvider = dataProviderCaptor.getValue();

    // 1. Fetch for task 1 -> should get entries 1 and 2.
    List<LogEntry> task1Entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(task1)))
                    .build())
            .get();
    assertThat(task1Entries).hasSize(2);

    // 2. Fetch for task 2 -> should ALSO get entries 1 and 2 (not yet contributed to task 2).
    List<LogEntry> task2Entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(task2)))
                    .build())
            .get();
    assertThat(task2Entries).hasSize(2);

    // 3. Log a 3rd entry.
    ActivityProto proto3 = ActivityProto.newBuilder().setDurationSeconds(300).build();
    logger.log(proto3);

    // 4. Fetch for task 1 again -> should ONLY get entry 3.
    List<LogEntry> task1SecondEntries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(task1)))
                    .build())
            .get();
    assertThat(task1SecondEntries).hasSize(1);
    assertThat(
            ActivityProto.parseFrom(
                    task1SecondEntries.get(0).value(), ExtensionRegistryLite.getEmptyRegistry())
                .getDurationSeconds())
        .isEqualTo(300);

    // 5. Fetch for task 2 again -> should ALSO ONLY get entry 3.
    List<LogEntry> task2SecondEntries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(task2)))
                    .build())
            .get();
    assertThat(task2SecondEntries).hasSize(1);
    assertThat(
            ActivityProto.parseFrom(
                    task2SecondEntries.get(0).value(), ExtensionRegistryLite.getEmptyRegistry())
                .getDurationSeconds())
        .isEqualTo(300);
  }

  @Test
  public void testUpload_partialRollbackWithMultipleTasks() throws Exception {
    String task1 = "task_success";
    String task2 = "task_failure";

    logger.log(ActivityProto.newBuilder().setDurationSeconds(100).build());

    // Start upload with unsettled future.
    SettableFuture<List<UploadOutcome>> connectorFuture = SettableFuture.create();
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(connectorFuture);
    ListenableFuture<Void> uploadFuture = logger.upload();
    DataProvider dataProvider = dataProviderCaptor.getValue();

    // Commit contributions for both tasks.
    List<LogEntry> task1Entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(task1)))
                    .build())
            .get();
    assertThat(task1Entries).hasSize(1);

    List<LogEntry> task2Entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(task2)))
                    .build())
            .get();
    assertThat(task2Entries).hasSize(1);

    // Complete upload: task 1 succeeded (CONTRIBUTED), task 2 failed (NOT_CONTRIBUTED).
    connectorFuture.set(
        ImmutableList.of(
            UploadOutcome.builder()
                .setTaskName(task1)
                .setStatus(UploadOutcome.Status.CONTRIBUTED)
                .build(),
            UploadOutcome.builder()
                .setTaskName(task2)
                .setStatus(UploadOutcome.Status.NOT_CONTRIBUTED)
                .build()));
    uploadFuture.get();

    // Start second upload.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused = logger.upload();
    DataProvider dataProvider2 = dataProviderCaptor.getValue();

    // Task 1 should find 0 entries (remained contributed).
    List<LogEntry> task1Entries2 =
        dataProvider2
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(task1)))
                    .build())
            .get();
    assertThat(task1Entries2).isEmpty();

    // Task 2 should find the entry again (rolled back).
    List<LogEntry> task2Entries2 =
        dataProvider2
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(task2)))
                    .build())
            .get();
    assertThat(task2Entries2).hasSize(1);
  }

  @Test
  public void testRollback_multipleEntries_allRolledBack() throws Exception {
    ActivityProto proto1 = ActivityProto.newBuilder().setDurationSeconds(100).build();
    ActivityProto proto2 = ActivityProto.newBuilder().setDurationSeconds(200).build();
    ActivityProto proto3 = ActivityProto.newBuilder().setDurationSeconds(300).build();
    logger.log(proto1);
    logger.log(proto2);
    logger.log(proto3);

    // Start upload 1.
    SettableFuture<List<UploadOutcome>> connectorFuture1 = SettableFuture.create();
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(connectorFuture1);
    ListenableFuture<Void> uploadFuture1 = logger.upload();
    DataProvider dataProvider1 = dataProviderCaptor.getValue();

    List<LogEntry> entries1 =
        dataProvider1
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();
    assertThat(entries1).hasSize(3);

    // Rollback all 3 entries.
    connectorFuture1.set(
        ImmutableList.of(
            UploadOutcome.builder()
                .setTaskName(TASK_NAME)
                .setStatus(UploadOutcome.Status.NOT_CONTRIBUTED)
                .build()));
    uploadFuture1.get();

    // Start upload 2 -> all 3 entries should be returned again.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused = logger.upload();
    DataProvider dataProvider2 = dataProviderCaptor.getValue();

    List<LogEntry> entries2 =
        dataProvider2
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();
    assertThat(entries2).hasSize(3);
  }

  @Test
  public void testRollback_unknownTask_doesNotThrowOrAffectOtherTasks() throws Exception {
    logger.log(ActivityProto.newBuilder().setDurationSeconds(100).build());

    SettableFuture<List<UploadOutcome>> connectorFuture = SettableFuture.create();
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(connectorFuture);
    ListenableFuture<Void> uploadFuture = logger.upload();
    DataProvider dataProvider = dataProviderCaptor.getValue();

    List<LogEntry> entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();
    assertThat(entries).hasSize(1);

    // Return outcome for an unknown task (never queried) marked NOT_CONTRIBUTED,
    // alongside TASK_NAME marked CONTRIBUTED.
    connectorFuture.set(
        ImmutableList.of(
            UploadOutcome.builder()
                .setTaskName("completely_unknown_task")
                .setStatus(UploadOutcome.Status.NOT_CONTRIBUTED)
                .build(),
            UploadOutcome.builder()
                .setTaskName(TASK_NAME)
                .setStatus(UploadOutcome.Status.CONTRIBUTED)
                .build()));
    uploadFuture.get();

    // Verify TASK_NAME entry remained contributed.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused = logger.upload();
    DataProvider dataProvider2 = dataProviderCaptor.getValue();

    List<LogEntry> entries2 =
        dataProvider2
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();
    assertThat(entries2).isEmpty();
  }

  @Test
  public void testMigration_fromSharedTracker_multipleTasksForSameLogSource() throws Exception {
    String task1 = "task_one";
    String task2 = "task_two";

    // Legacy centralized task contribution tracker DB: entry 1 contributed to task1, entry 2
    // contributed to task2.
    createLegacyCentralizedTaskContributionTrackerDatabaseWithTasks(
        TEST_PRIVATE_LOG_SOURCE,
        ImmutableMap.of(
            task1, ImmutableList.of(1L),
            task2, ImmutableList.of(2L)));

    context.deleteDatabase(PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE));
    logger = createLogger();

    // Log entries 1, 2, 3 into new logger.
    logger.log(ActivityProto.newBuilder().setDurationSeconds(100).build()); // id=1
    logger.log(ActivityProto.newBuilder().setDurationSeconds(200).build()); // id=2
    logger.log(ActivityProto.newBuilder().setDurationSeconds(300).build()); // id=3

    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused = logger.upload();
    DataProvider dataProvider = dataProviderCaptor.getValue();

    // Task 1 should filter out entry 1, returning entries 2 and 3.
    List<LogEntry> task1Entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(task1)))
                    .build())
            .get();
    assertThat(task1Entries).hasSize(2);
    assertThat(
            ActivityProto.parseFrom(
                    task1Entries.get(0).value(), ExtensionRegistryLite.getEmptyRegistry())
                .getDurationSeconds())
        .isEqualTo(200);
    assertThat(
            ActivityProto.parseFrom(
                    task1Entries.get(1).value(), ExtensionRegistryLite.getEmptyRegistry())
                .getDurationSeconds())
        .isEqualTo(300);

    // Task 2 should filter out entry 2, returning entries 1 and 3.
    List<LogEntry> task2Entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(task2)))
                    .build())
            .get();
    assertThat(task2Entries).hasSize(2);
    assertThat(
            ActivityProto.parseFrom(
                    task2Entries.get(0).value(), ExtensionRegistryLite.getEmptyRegistry())
                .getDurationSeconds())
        .isEqualTo(100);
    assertThat(
            ActivityProto.parseFrom(
                    task2Entries.get(1).value(), ExtensionRegistryLite.getEmptyRegistry())
                .getDurationSeconds())
        .isEqualTo(300);
  }

  @Test
  public void testMigration_whenOldDbDoesNotExist_marksMigrationCompleted() throws Exception {
    File dbFile = context.getDatabasePath("task-contribution-database");
    if (dbFile.exists()) {
      dbFile.delete();
    }
    context.deleteDatabase(PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE));

    logger = createLogger();
    // Block on migration.
    ((PrivateLoggerImpl) logger).migrateFromSharedTracker().get();

    // Verify metadata was written.
    PrivateLoggerDatabase db =
        Room.databaseBuilder(
                context,
                PrivateLoggerDatabase.class,
                PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE))
            .allowMainThreadQueries()
            .build();
    assertThat(db.databaseMetadataDao().hasKey("shared_tracker_migrated")).isTrue();
    assertThat(db.databaseMetadataDao().getValue("shared_tracker_migrated")).isEqualTo("true");
    db.close();
  }

  @Test
  public void testCleanupOldEntries_prunesOrphanTasks() throws Exception {
    Duration retention = Duration.ofDays(1);
    PrivateLoggerConfig config =
        PrivateLoggerConfig.builder()
            .setPrivateLogSourceName(TEST_PRIVATE_LOG_SOURCE)
            .setRetentionPeriod(retention)
            .build();
    logger =
        PrivateLoggerImpl.create(
            PrivateLoggerImplDeps.newBuilder()
                .setContext(context)
                .setConfig(config)
                .setConnector(mockConnector)
                .setDbExecutor(executor)
                .setTimeSource(fakeTimeSource)
                .setStabilityMonitor(mockStabilityMonitor)
                .build());

    // 1. Log and contribute entry 1 to task "task1".
    ActivityProto proto1 = ActivityProto.newBuilder().setDurationSeconds(100).build();
    logger.log(proto1);
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    logger.upload().get();
    dataProviderCaptor
        .getValue()
        .getData(
            SelectorContext.newBuilder()
                .setComputationProperties(
                    QueryTimeComputationProperties.newBuilder()
                        .setFederated(FederatedComputation.newBuilder().setTaskName("task1")))
                .build())
        .get();

    // 2. Advance time past retention period (2 days).
    fakeTimeSource.advance(Duration.ofDays(2));

    // 3. Log and contribute entry 2 to task "task2".
    ActivityProto proto2 = ActivityProto.newBuilder().setDurationSeconds(200).build();
    logger.log(proto2);
    dataProviderCaptor
        .getValue()
        .getData(
            SelectorContext.newBuilder()
                .setComputationProperties(
                    QueryTimeComputationProperties.newBuilder()
                        .setFederated(FederatedComputation.newBuilder().setTaskName("task2")))
                .build())
        .get();

    // Verify both tasks exist before cleanup.
    PrivateLoggerDatabase db =
        Room.databaseBuilder(
                context,
                PrivateLoggerDatabase.class,
                PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE))
            .allowMainThreadQueries()
            .build();
    assertThat(db.taskDao().getTaskId("task1")).isNotNull();
    assertThat(db.taskDao().getTaskId("task2")).isNotNull();
    db.close();

    // 4. Trigger retention cleanup.
    ((PrivateLoggerImpl) logger).cleanupOldEntries().get();

    // 5. Verify task1 is pruned because its only contribution (entry 1) was older than retention
    // and deleted, while task2 is preserved because its contribution is still within retention.
    db =
        Room.databaseBuilder(
                context,
                PrivateLoggerDatabase.class,
                PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE))
            .allowMainThreadQueries()
            .build();
    assertThat(db.taskDao().getTaskId("task1")).isNull();
    assertThat(db.taskDao().getTaskId("task2")).isNotNull();
    db.close();
  }

  @Test
  public void testTaskDao_deleteOrphanTasks() {
    PrivateLoggerDatabase db =
        Room.databaseBuilder(
                context,
                PrivateLoggerDatabase.class,
                PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE))
            .allowMainThreadQueries()
            .build();

    Task task1 = new Task();
    task1.taskName = "taskWithContributions";
    long task1Id = db.taskDao().insert(task1);

    Task task2 = new Task();
    task2.taskName = "orphanTask";
    long task2Id = db.taskDao().insert(task2);

    ContributedEntry entry = new ContributedEntry();
    entry.taskId = task1Id;
    entry.entryId = 100L;
    entry.insertionTimeMillis = 1000L;
    db.contributedEntryDao().insertAll(ImmutableList.of(entry));

    assertThat(db.taskDao().getTaskId("taskWithContributions")).isEqualTo(task1Id);
    assertThat(db.taskDao().getTaskId("orphanTask")).isEqualTo(task2Id);

    int deletedCount = db.taskDao().deleteOrphanTasks();
    assertThat(deletedCount).isEqualTo(1);
    assertThat(db.taskDao().getTaskId("taskWithContributions")).isEqualTo(task1Id);
    assertThat(db.taskDao().getTaskId("orphanTask")).isNull();

    db.close();
  }

  @Test
  public void testTaskDao_getOrCreateTaskId_whenNewTask_createsAndReturnsId() {
    PrivateLoggerDatabase db =
        Room.databaseBuilder(
                context,
                PrivateLoggerDatabase.class,
                PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE))
            .allowMainThreadQueries()
            .build();

    long newId = db.taskDao().getOrCreateTaskId("newTask");
    assertThat(newId).isGreaterThan(0L);
    assertThat(db.taskDao().getTaskId("newTask")).isEqualTo(newId);

    db.close();
  }

  @Test
  public void testTaskDao_getOrCreateTaskId_whenAlreadyExists_returnsExistingId() {
    PrivateLoggerDatabase db =
        Room.databaseBuilder(
                context,
                PrivateLoggerDatabase.class,
                PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE))
            .allowMainThreadQueries()
            .build();

    // 1. Pre-insert the task directly so it already exists.
    Task task = new Task();
    task.taskName = "existingTask";
    long initialId = db.taskDao().insert(task);
    assertThat(initialId).isGreaterThan(0L);

    // 2. getOrCreateTaskId will attempt insert, get -1 due to UNIQUE constraint,
    // and fall back to getTaskId(taskName).
    long retrievedId = db.taskDao().getOrCreateTaskId("existingTask");
    assertThat(retrievedId).isEqualTo(initialId);

    db.close();
  }

  @Test
  public void
      testContributedEntryDao_markEntriesAsContributedForTimeRange_filtersByCutoffAndTask() {
    PrivateLoggerDatabase db =
        Room.databaseBuilder(
                context,
                PrivateLoggerDatabase.class,
                PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE))
            .allowMainThreadQueries()
            .build();

    long task1Id = db.taskDao().getOrCreateTaskId("task1");
    long task2Id = db.taskDao().getOrCreateTaskId("task2");

    DbLogEntry oldEntry = new DbLogEntry();
    oldEntry.value = new byte[] {1};
    oldEntry.timestampMillis = 1000L;
    oldEntry.timestamp = "2026-09-16T00:00:01Z";

    DbLogEntry cutoffBoundaryEntry = new DbLogEntry();
    cutoffBoundaryEntry.value = new byte[] {2};
    cutoffBoundaryEntry.timestampMillis = 2000L;
    cutoffBoundaryEntry.timestamp = "2026-09-16T00:00:02Z";

    DbLogEntry newEntry1 = new DbLogEntry();
    newEntry1.value = new byte[] {3};
    newEntry1.timestampMillis = 3000L;
    newEntry1.timestamp = "2026-09-16T00:00:03Z";

    DbLogEntry newEntry2 = new DbLogEntry();
    newEntry2.value = new byte[] {4};
    newEntry2.timestampMillis = 4000L;
    newEntry2.timestamp = "2026-09-16T00:00:04Z";

    db.logEntryDao()
        .insertAll(ImmutableList.of(oldEntry, cutoffBoundaryEntry, newEntry1, newEntry2));
    // Query with an unused taskId (-1L) to retrieve all inserted entries via production API.
    List<DbLogEntry> insertedEntries = db.logEntryDao().getUncontributedEntries(-1L, 0L);
    assertThat(insertedEntries).hasSize(4);
    long oldEntryId = insertedEntries.get(0).id;
    long cutoffBoundaryEntryId = insertedEntries.get(1).id;
    long newEntry1Id = insertedEntries.get(2).id;
    long newEntry2Id = insertedEntries.get(3).id;

    // Pre-mark newEntry1 as contributed to task2; it should still be uncontributed for task1.
    ContributedEntry task2Contrib = new ContributedEntry();
    task2Contrib.taskId = task2Id;
    task2Contrib.entryId = newEntry1Id;
    task2Contrib.insertionTimeMillis = 2500L;
    db.contributedEntryDao().insertAll(ImmutableList.of(task2Contrib));

    // Mark uncontributed entries for task1 with cutoff = 2000L (strictly greater than 2000L).
    db.contributedEntryDao().markEntriesAsContributedForTimeRange(task1Id, 2000L, 5000L);

    // For task1, newEntry1 and newEntry2 are now contributed, leaving only oldEntry and
    // cutoffBoundaryEntry uncontributed when querying with cutoff 0L.
    assertThat(db.logEntryDao().getUncontributedEntries(task1Id, 0L).stream().map(e -> (long) e.id))
        .containsExactly(oldEntryId, cutoffBoundaryEntryId);
    assertThat(db.logEntryDao().getUncontributedEntries(task1Id, 2000L)).isEmpty();

    // For task2, only newEntry1 is contributed, so newEntry2 remains uncontributed.
    assertThat(db.logEntryDao().getUncontributedEntries(task2Id, 0L).stream().map(e -> (long) e.id))
        .containsExactly(oldEntryId, cutoffBoundaryEntryId, newEntry2Id);

    db.close();
  }

  @Test
  public void
      testContributedEntryDao_markEntriesAsContributedForTimeRange_preservesExistingContributionsAndIsIdempotent() {
    PrivateLoggerDatabase db =
        Room.databaseBuilder(
                context,
                PrivateLoggerDatabase.class,
                PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE))
            .allowMainThreadQueries()
            .build();

    long taskId = db.taskDao().getOrCreateTaskId("task1");

    DbLogEntry entry1 = new DbLogEntry();
    entry1.value = new byte[] {1};
    entry1.timestampMillis = 3000L;
    entry1.timestamp = "2026-09-16T00:00:03Z";

    DbLogEntry entry2 = new DbLogEntry();
    entry2.value = new byte[] {2};
    entry2.timestampMillis = 4000L;
    entry2.timestamp = "2026-09-16T00:00:04Z";

    db.logEntryDao().insertAll(ImmutableList.of(entry1, entry2));
    List<DbLogEntry> insertedEntries = db.logEntryDao().getUncontributedEntries(-1L, 0L);
    long entry1Id = insertedEntries.get(0).id;

    // Pre-insert entry1 with an older insertionTimeMillis (1500L).
    ContributedEntry existingContrib = new ContributedEntry();
    existingContrib.taskId = taskId;
    existingContrib.entryId = entry1Id;
    existingContrib.insertionTimeMillis = 1500L;
    db.contributedEntryDao().insertAll(ImmutableList.of(existingContrib));

    // Run markEntriesAsContributedForTimeRange with insertionTimeMillis = 5000L.
    db.contributedEntryDao().markEntriesAsContributedForTimeRange(taskId, 2000L, 5000L);
    // Run a second time to verify idempotency.
    db.contributedEntryDao().markEntriesAsContributedForTimeRange(taskId, 2000L, 6000L);

    // Both entry1 and entry2 are contributed, so 0 entries are uncontributed.
    assertThat(db.logEntryDao().getUncontributedEntries(taskId, 0L)).isEmpty();

    // Deleting contributions older than 2000L should remove entry1 (whose insertionTimeMillis
    // was preserved at 1500L) while retaining entry2 (whose insertionTimeMillis is 5000L).
    db.contributedEntryDao().deleteOlderThan(2000L);
    assertThat(db.logEntryDao().getUncontributedEntries(taskId, 0L).stream().map(e -> (long) e.id))
        .containsExactly(entry1Id);

    db.close();
  }

  @Test
  public void testUpload_with2000Entries_batchesAndRollsBackAcrossChunkBoundaries()
      throws Exception {
    context.deleteDatabase(PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE));
    logger = createLogger();

    // 1. Log 2,000 entries.
    for (int i = 0; i < 2000; i++) {
      ActivityProto proto = ActivityProto.newBuilder().setDurationSeconds(i).build();
      logger.log(proto);
    }

    // 2. Start upload 1: fetch all 2,000 entries.
    SettableFuture<List<UploadOutcome>> connectorFuture1 = SettableFuture.create();
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(connectorFuture1);
    ListenableFuture<Void> uploadFuture1 = logger.upload();
    DataProvider dataProvider1 = dataProviderCaptor.getValue();

    List<LogEntry> entries1 =
        dataProvider1
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();
    assertThat(entries1).hasSize(2000);

    // 3. Rollback the upload with NOT_CONTRIBUTED. This forces rollbackContributions to partition
    // the 2,000 entry IDs across multiple chunks of 900 (900, 900, 200).
    connectorFuture1.set(
        ImmutableList.of(
            UploadOutcome.builder()
                .setTaskName(TASK_NAME)
                .setStatus(UploadOutcome.Status.NOT_CONTRIBUTED)
                .build()));
    uploadFuture1.get();

    // 4. Start upload 2: verify all 2,000 entries were successfully rolled back and are available.
    SettableFuture<List<UploadOutcome>> connectorFuture2 = SettableFuture.create();
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(connectorFuture2);
    ListenableFuture<Void> uploadFuture2 = logger.upload();
    DataProvider dataProvider2 = dataProviderCaptor.getValue();

    List<LogEntry> entries2 =
        dataProvider2
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();
    assertThat(entries2).hasSize(2000);

    // 5. Complete upload 2 with CONTRIBUTED.
    connectorFuture2.set(
        ImmutableList.of(
            UploadOutcome.builder()
                .setTaskName(TASK_NAME)
                .setStatus(UploadOutcome.Status.CONTRIBUTED)
                .build()));
    uploadFuture2.get();

    // 6. Start upload 3: verify 0 entries remain uncontributed.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused = logger.upload();
    DataProvider dataProvider3 = dataProviderCaptor.getValue();

    List<LogEntry> entries3 =
        dataProvider3
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();
    assertThat(entries3).isEmpty();
  }

  @Test
  public void testMigration_withLargeNumberOfContributions_migratesInBatches() throws Exception {
    // 1. Create legacy centralized task contribution tracker DB with 2,500 contribution records
    // (exceeds BATCH_SIZE = 1000).
    ImmutableList.Builder<Long> legacyEntryIds = ImmutableList.builder();
    for (long i = 1; i <= 2500; i++) {
      legacyEntryIds.add(i);
    }
    createLegacyCentralizedTaskContributionTrackerDatabase(
        TEST_PRIVATE_LOG_SOURCE, TASK_NAME, legacyEntryIds.build());

    context.deleteDatabase(PrivateLoggerImpl.getDatabaseName(TEST_PRIVATE_LOG_SOURCE));
    logger = createLogger();

    // Block on migration to ensure all batches complete.
    ((PrivateLoggerImpl) logger).migrateFromSharedTracker().get();

    // 2. Log 2,500 entries matching the legacy IDs (1..2500) plus 1 new entry (2501).
    for (int i = 0; i < 2501; i++) {
      ActivityProto proto = ActivityProto.newBuilder().setDurationSeconds(i).build();
      logger.log(proto);
    }

    // 3. Retrieve data for the task. All 2,500 migrated entries should be filtered out,
    // returning only the 1 newly logged entry.
    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused = logger.upload();

    DataProvider dataProvider = dataProviderCaptor.getValue();
    List<LogEntry> entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();

    assertThat(entries).hasSize(1);
    assertThat(
            ActivityProto.parseFrom(
                    entries.get(0).value(), ExtensionRegistryLite.getEmptyRegistry())
                .getDurationSeconds())
        .isEqualTo(2500);
  }

  private void createLegacyCentralizedTaskContributionTrackerDatabase(
      String logSource, String taskName, List<Long> entryIds) {
    createLegacyCentralizedTaskContributionTrackerDatabaseWithTasks(
        logSource, ImmutableMap.of(taskName, entryIds));
  }

  private void createLegacyCentralizedTaskContributionTrackerDatabaseWithTasks(
      String logSource, Map<String, List<Long>> taskContributions) {
    File dbFile = context.getDatabasePath("task-contribution-database");
    dbFile.getParentFile().mkdirs();
    if (dbFile.exists()) {
      dbFile.delete();
    }
    try (SQLiteDatabase db = SQLiteDatabase.openOrCreateDatabase(dbFile, null)) {
      db.execSQL(
          "CREATE TABLE IF NOT EXISTS Task (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
              + " logSource TEXT, taskName TEXT)");
      db.execSQL(
          "CREATE TABLE IF NOT EXISTS EntryContribution (id INTEGER PRIMARY KEY AUTOINCREMENT NOT"
              + " NULL, taskId INTEGER NOT NULL, valueId INTEGER NOT NULL, insertionTimeMillis"
              + " INTEGER NOT NULL)");

      db.beginTransaction();
      try {
        int taskId = 1;
        for (Map.Entry<String, List<Long>> entry : taskContributions.entrySet()) {
          db.execSQL(
              "INSERT INTO Task (id, logSource, taskName) VALUES (?, ?, ?)",
              new Object[] {taskId, logSource, entry.getKey()});
          for (long entryId : entry.getValue()) {
            db.execSQL(
                "INSERT INTO EntryContribution (taskId, valueId, insertionTimeMillis) VALUES (?, ?,"
                    + " ?)",
                new Object[] {taskId, entryId, fakeTimeSource.now().toEpochMilli()});
          }
          taskId++;
        }
        db.setTransactionSuccessful();
      } finally {
        db.endTransaction();
      }
    }
  }

  @Test
  public void testLog_list_insertsAllEntries() throws Exception {
    TimeZone.setDefault(TimeZone.getTimeZone("America/New_York"));
    Instant logTime = fakeTimeSource.now();

    ImmutableList<ActivityProto> testProtos =
        ImmutableList.of(
            ActivityProto.newBuilder().setDurationSeconds(10).build(),
            ActivityProto.newBuilder().setDurationSeconds(20).build(),
            ActivityProto.newBuilder().setDurationSeconds(30).build());

    logger.log(testProtos);

    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused = logger.upload();

    DataProvider dataProvider = dataProviderCaptor.getValue();
    List<LogEntry> entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();

    assertThat(entries).hasSize(3);
    String expectedTimestamp = TimestampUtil.getIso8601String(logTime);
    for (int i = 0; i < 3; i++) {
      LogEntry entry = entries.get(i);
      ActivityProto proto =
          ActivityProto.parseFrom(entry.value(), ExtensionRegistryLite.getEmptyRegistry());
      assertThat(proto.getDurationSeconds()).isEqualTo((i + 1) * 10);
      assertThat(entry.timestamp()).isEqualTo(expectedTimestamp);
    }
  }

  @Test
  public void testLog_list_emptyAndNull_doesNotFail() throws Exception {
    logger.log((List<ActivityProto>) null);
    logger.log(ImmutableList.of());

    when(mockConnector.upload(eq(TEST_PRIVATE_LOG_SOURCE), dataProviderCaptor.capture()))
        .thenReturn(immediateFuture(ImmutableList.of()));
    var unused = logger.upload();

    DataProvider dataProvider = dataProviderCaptor.getValue();
    List<LogEntry> entries =
        dataProvider
            .getData(
                SelectorContext.newBuilder()
                    .setComputationProperties(
                        QueryTimeComputationProperties.newBuilder()
                            .setFederated(FederatedComputation.newBuilder().setTaskName(TASK_NAME)))
                    .build())
            .get();

    assertThat(entries).isEmpty();
  }

  @Test
  public void testLog_list_exceptionReportedToStabilityMonitor() throws Exception {
    logger.log(Arrays.asList((ActivityProto) null));

    verify(mockStabilityMonitor).reportCrash(any(NullPointerException.class));
  }
}
