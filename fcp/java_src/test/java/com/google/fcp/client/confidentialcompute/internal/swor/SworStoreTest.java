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

import androidx.datastore.guava.GuavaDataStore;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.common.collect.ImmutableMap;
import java.io.File;
import java.nio.file.Files;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.Optional;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public final class SworStoreTest {

  private static final String KEY = "workload";
  private static final Duration SWOR_PERIOD = Duration.ofHours(24);
  private static final Instant T0 = Instant.ofEpochMilli(1_000_000L);

  @Rule public final TemporaryFolder temporaryFolder = new TemporaryFolder();

  private SworStore sworStore;

  @Before
  public void setUp() throws Exception {
    File file = new File(temporaryFolder.getRoot(), SworStore.FILE_NAME);
    sworStore =
        new SworStore(new GuavaDataStore.Builder<>(SworStore.serializer(), () -> file).build());
    // Makes the records trustworthy from well before T0, so that only the records decide.
    var unused = isWithin(sworStore, KEY, SWOR_PERIOD, T0.minus(Duration.ofDays(30)));
  }

  @Test
  public void keysWithinSworPeriod_freshStore_returnsTrueUntilPeriodSinceFirstUse()
      throws Exception {
    File file = new File(temporaryFolder.getRoot(), "fresh_" + SworStore.FILE_NAME);
    SworStore store =
        new SworStore(new GuavaDataStore.Builder<>(SworStore.serializer(), () -> file).build());

    assertThat(isWithin(store, KEY, SWOR_PERIOD, T0)).isTrue();
    assertThat(isWithin(store, KEY, SWOR_PERIOD, T0.plus(SWOR_PERIOD).minusMillis(1))).isTrue();
    assertThat(isWithin(store, KEY, SWOR_PERIOD, T0.plus(SWOR_PERIOD))).isFalse();
  }

  @Test
  public void keysWithinSworPeriod_freshStoreAndZeroPeriod_returnsFalse() throws Exception {
    File file = new File(temporaryFolder.getRoot(), "fresh_" + SworStore.FILE_NAME);
    SworStore store =
        new SworStore(new GuavaDataStore.Builder<>(SworStore.serializer(), () -> file).build());

    assertThat(isWithin(store, KEY, Duration.ZERO, T0)).isFalse();
  }

  @Test
  public void keysWithinSworPeriod_noRecord_returnsFalse() throws Exception {
    assertThat(isWithin(sworStore, KEY, SWOR_PERIOD, T0)).isFalse();
    assertThat(sworStore.lastUpload(KEY).get()).isEmpty();
  }

  @Test
  public void keysWithinSworPeriod_withinPeriod_returnsTrue() throws Exception {
    sworStore.recordUpload(KEY, T0).get();
    assertThat(sworStore.lastUpload(KEY).get()).hasValue(T0);

    Instant oneHourLater = T0.plus(Duration.ofHours(1));
    assertThat(isWithin(sworStore, KEY, SWOR_PERIOD, oneHourLater)).isTrue();
  }

  @Test
  public void keysWithinSworPeriod_atPeriodBoundary_returnsTrue() throws Exception {
    sworStore.recordUpload(KEY, T0).get();

    Instant exactlyPeriodLater = T0.plus(SWOR_PERIOD);
    assertThat(isWithin(sworStore, KEY, SWOR_PERIOD, exactlyPeriodLater)).isTrue();
  }

  @Test
  public void keysWithinSworPeriod_afterPeriod_returnsFalse() throws Exception {
    sworStore.recordUpload(KEY, T0).get();

    Instant afterPeriod = T0.plus(SWOR_PERIOD).plusMillis(1);
    assertThat(isWithin(sworStore, KEY, SWOR_PERIOD, afterPeriod)).isFalse();
  }

  @Test
  public void keysWithinSworPeriod_recordInTheFuture_returnsTrue() throws Exception {
    sworStore.recordUpload(KEY, T0.plus(Duration.ofDays(5)).plusMillis(1)).get();

    assertThat(isWithin(sworStore, KEY, SWOR_PERIOD, T0)).isTrue();
  }

  @Test
  public void keysWithinSworPeriod_differentKey_returnsFalse() throws Exception {
    sworStore.recordUpload(KEY, T0).get();

    assertThat(isWithin(sworStore, "other/key", SWOR_PERIOD, T0)).isFalse();
  }

  @Test
  public void recordUpload_noPrevious_returnsEmpty() throws Exception {
    assertThat(sworStore.recordUpload(KEY, T0).get()).isEmpty();
    assertThat(sworStore.lastUpload(KEY).get()).hasValue(T0);
  }

  @Test
  public void recordUpload_withPrevious_returnsPrevious() throws Exception {
    sworStore.recordUpload(KEY, T0).get();
    Instant later = T0.plus(Duration.ofDays(2));

    assertThat(sworStore.recordUpload(KEY, later).get()).hasValue(T0);
    assertThat(sworStore.lastUpload(KEY).get()).hasValue(later);
  }

  @Test
  public void restoreUpload_withPrevious_putsPreviousBack() throws Exception {
    sworStore.recordUpload(KEY, T0).get();
    Instant recorded = T0.plus(Duration.ofDays(2));
    Optional<Instant> previous = sworStore.recordUpload(KEY, recorded).get();

    sworStore.restoreUpload(KEY, recorded, previous).get();

    assertThat(sworStore.lastUpload(KEY).get()).hasValue(T0);
  }

  @Test
  public void restoreUpload_noPrevious_removesRecord() throws Exception {
    Optional<Instant> previous = sworStore.recordUpload(KEY, T0).get();

    sworStore.restoreUpload(KEY, T0, previous).get();

    assertThat(sworStore.lastUpload(KEY).get()).isEmpty();
    assertThat(isWithin(sworStore, KEY, SWOR_PERIOD, T0)).isFalse();
  }

  @Test
  public void restoreUpload_storedTimeDoesNotMatchRecorded_leavesStoreAlone() throws Exception {
    Instant firstUpload = T0;
    Optional<Instant> firstPrevious = sworStore.recordUpload(KEY, firstUpload).get();

    Instant secondUpload = T0.plus(Duration.ofHours(1));
    var unused = sworStore.recordUpload(KEY, secondUpload).get();

    // First upload aborts after second upload recorded. The store now holds secondUpload,
    // so restoring the first upload leaves the store alone.
    sworStore.restoreUpload(KEY, firstUpload, firstPrevious).get();

    assertThat(sworStore.lastUpload(KEY).get()).hasValue(secondUpload);
  }

  @Test
  public void corruptFile_isReplacedWithEmptyStateTrustworthyFromFirstUse() throws Exception {
    File file = temporaryFolder.newFile("corrupt_" + SworStore.FILE_NAME);
    byte[] notAProto = new byte[12];
    Arrays.fill(notAProto, (byte) 0xff);
    Files.write(file.toPath(), notAProto);
    SworStore store =
        new SworStore(
            new GuavaDataStore.Builder<>(SworStore.serializer(), () -> file)
                .setCorruptionHandler(SworStore.corruptionHandler())
                .build());

    // The replaced state is trustworthy only from its first use.
    assertThat(isWithin(store, KEY, SWOR_PERIOD, T0)).isTrue();
    assertThat(isWithin(store, KEY, SWOR_PERIOD, T0.plus(SWOR_PERIOD))).isFalse();
    assertThat(store.recordUpload(KEY, T0).get()).isEmpty();
    assertThat(store.lastUpload(KEY).get()).hasValue(T0);
  }

  @Test
  public void keysWithinSworPeriod_returnsOnlyKeysWithinTheirPeriod() throws Exception {
    sworStore.recordUpload(KEY, T0).get();

    assertThat(
            sworStore
                .keysWithinSworPeriod(
                    ImmutableMap.of(KEY, SWOR_PERIOD, "other", SWOR_PERIOD), T0.plusSeconds(1))
                .get())
        .containsExactly(KEY);
  }

  @Test
  public void keysWithinSworPeriod_dropsRecordsOlderThanRetention() throws Exception {
    sworStore.recordUpload(KEY, T0).get();
    Instant afterRetention = T0.plus(SworStore.RETENTION).plusMillis(1);

    assertThat(isWithin(sworStore, KEY, SWOR_PERIOD, afterRetention)).isFalse();

    assertThat(sworStore.lastUpload(KEY).get()).isEmpty();
  }

  @Test
  public void forContext_storesShareOneFile() throws Exception {
    SworStore first = SworStore.forContext(ApplicationProvider.getApplicationContext());
    SworStore second = SworStore.forContext(ApplicationProvider.getApplicationContext());

    first.recordUpload(KEY, T0).get();

    assertThat(second.lastUpload(KEY).get()).hasValue(T0);
  }

  private static boolean isWithin(SworStore store, String key, Duration sworPeriod, Instant now)
      throws Exception {
    return store.keysWithinSworPeriod(ImmutableMap.of(key, sworPeriod), now).get().contains(key);
  }
}
