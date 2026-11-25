// Copyright 2026 Google LLC
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
package com.google.fcp.client.confidentialcompute.internal.swor;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import androidx.datastore.guava.GuavaDataStore;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.common.collect.ImmutableMap;
import com.google.fcp.client.FakeTimeSource;
import com.google.fcp.client.confidentialcompute.UploadAbortedException;
import com.google.fcp.client.confidentialcompute.internal.swor.SworTracker.SworRecord;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public final class SworTrackerTest {

  private static final String WORKLOAD_SET = "workload";
  private static final Duration SWOR_PERIOD = Duration.ofHours(24);

  @Rule public final TemporaryFolder temporaryFolder = new TemporaryFolder();

  private final FakeTimeSource timeSource = new FakeTimeSource();
  private SworStore sworStore;
  private SworTracker tracker;

  @Before
  public void setUp() {
    sworStore = sworStore(new File(temporaryFolder.getRoot(), SworStore.FILE_NAME));
    tracker = new SworTracker(sworStore, timeSource);
  }

  @Test
  public void workloadSetsWithinPeriod_freshStore_returnsTrueUntilPeriodSinceFirstUse()
      throws Exception {
    assertThat(isWithin(tracker, WORKLOAD_SET, SWOR_PERIOD)).isTrue();

    timeSource.advance(SWOR_PERIOD);
    assertThat(isWithin(tracker, WORKLOAD_SET, SWOR_PERIOD)).isFalse();
  }

  @Test
  public void workloadSetsWithinPeriod_afterRecordUpload_returnsTrueUntilPeriodEnds()
      throws Exception {
    var unused = tracker.recordUpload(WORKLOAD_SET);

    timeSource.advance(SWOR_PERIOD);
    assertThat(isWithin(tracker, WORKLOAD_SET, SWOR_PERIOD)).isTrue();
    timeSource.advance(Duration.ofMillis(1));
    assertThat(isWithin(tracker, WORKLOAD_SET, SWOR_PERIOD)).isFalse();
  }

  @Test
  public void workloadSetsWithinPeriod_storeReadFails_returnsTrue() throws Exception {
    SworTracker failing = new SworTracker(unreadableSworStore(), timeSource);

    assertThat(isWithin(failing, WORKLOAD_SET, SWOR_PERIOD)).isTrue();
  }

  @Test
  public void recordUpload_recordsCurrentTime() throws Exception {
    var unused = tracker.recordUpload(WORKLOAD_SET);

    assertThat(sworStore.lastUpload(WORKLOAD_SET).get()).hasValue(timeSource.now());
  }

  @Test
  public void restoreUpload_noPreviousRecord_removesRecord() throws Exception {
    SworRecord record = tracker.recordUpload(WORKLOAD_SET);

    tracker.restoreUpload(record);

    assertThat(sworStore.lastUpload(WORKLOAD_SET).get()).isEmpty();
  }

  @Test
  public void restoreUpload_previousRecord_putsItBack() throws Exception {
    Instant first = timeSource.now();
    var unused = tracker.recordUpload(WORKLOAD_SET);
    timeSource.advance(Duration.ofDays(2));
    SworRecord record = tracker.recordUpload(WORKLOAD_SET);

    tracker.restoreUpload(record);

    assertThat(sworStore.lastUpload(WORKLOAD_SET).get()).hasValue(first);
  }

  @Test
  public void restoreUpload_storedTimeDoesNotMatchRecorded_leavesStoreAlone() throws Exception {
    SworRecord firstRecord = tracker.recordUpload(WORKLOAD_SET);

    timeSource.advance(Duration.ofHours(1));
    Instant second = timeSource.now();
    var unused = tracker.recordUpload(WORKLOAD_SET);

    tracker.restoreUpload(firstRecord);

    assertThat(sworStore.lastUpload(WORKLOAD_SET).get()).hasValue(second);
  }

  @Test
  public void recordUpload_storeWriteFails_throwsUploadAbortedException() throws Exception {
    SworTracker failing = new SworTracker(unwritableSworStore(), timeSource);

    assertThrows(UploadAbortedException.class, () -> failing.recordUpload(WORKLOAD_SET));
  }

  @Test
  public void disabled_skipsNothingAndRecordsNothing() throws Exception {
    SworRecord record = SworTracker.DISABLED.recordUpload(WORKLOAD_SET);
    SworTracker.DISABLED.restoreUpload(record);

    assertThat(isWithin(SworTracker.DISABLED, WORKLOAD_SET, SWOR_PERIOD)).isFalse();
  }

  private static SworStore sworStore(File file) {
    return new SworStore(new GuavaDataStore.Builder<>(SworStore.serializer(), () -> file).build());
  }

  /**
   * A store whose reads fail: its file holds bytes that are not a valid proto and, unlike the store
   * from {@link SworStore#forContext}, it has no corruption handler.
   */
  private SworStore unreadableSworStore() throws IOException {
    File file = temporaryFolder.newFile("unreadable_swor_store.pb");
    byte[] notAProto = new byte[12];
    Arrays.fill(notAProto, (byte) 0xff);
    Files.write(file.toPath(), notAProto);
    return sworStore(file);
  }

  /** A store whose writes fail: its parent "directory" is a regular file. */
  private SworStore unwritableSworStore() throws IOException {
    File notADirectory = temporaryFolder.newFile("not_a_directory");
    return sworStore(new File(notADirectory, "swor_store.pb"));
  }

  @Test
  public void workloadSetsWithinPeriod_returnsOnlySetsWithinTheirPeriod() throws Exception {
    // The first check makes the store's records trustworthy from now on.
    var unused1 = isWithin(tracker, WORKLOAD_SET, SWOR_PERIOD);
    timeSource.advance(SWOR_PERIOD.multipliedBy(2));
    var unused2 = tracker.recordUpload(WORKLOAD_SET);

    assertThat(
            tracker.workloadSetsWithinPeriod(
                ImmutableMap.of(WORKLOAD_SET, SWOR_PERIOD, "other", SWOR_PERIOD)))
        .containsExactly(WORKLOAD_SET);
  }

  @Test
  public void workloadSetsWithinPeriod_storeReadFails_returnsAllSets() throws Exception {
    SworTracker failing = new SworTracker(unreadableSworStore(), timeSource);

    assertThat(
            failing.workloadSetsWithinPeriod(
                ImmutableMap.of(WORKLOAD_SET, SWOR_PERIOD, "other", SWOR_PERIOD)))
        .containsExactly(WORKLOAD_SET, "other");
  }

  private static boolean isWithin(SworTracker tracker, String workloadSet, Duration sworPeriod)
      throws Exception {
    return tracker
        .workloadSetsWithinPeriod(ImmutableMap.of(workloadSet, sworPeriod))
        .contains(workloadSet);
  }
}
