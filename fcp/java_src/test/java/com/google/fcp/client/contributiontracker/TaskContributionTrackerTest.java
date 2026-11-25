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
package com.google.fcp.client.contributiontracker;

import static com.google.common.truth.Truth.assertThat;

import android.content.Context;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.fcp.client.FakeTimeSource;
import com.google.fcp.client.stabilitymonitor.StabilityMonitor;
import java.time.Duration;
import java.util.concurrent.Executors;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

@RunWith(AndroidJUnit4.class)
public final class TaskContributionTrackerTest {
  @Rule public final MockitoRule mocks = MockitoJUnit.rule();

  private static final String LOG_SOURCE_1 = "pop1";
  private static final String LOG_SOURCE_2 = "pop2";
  private static final String TASK_NAME_1 = "task1";
  private static final String TASK_NAME_2 = "task2";
  private static final long ENTRY_ID_1 = 1L;
  private static final long ENTRY_ID_2 = 2L;
  private static final long ENTRY_ID_3 = 3L;
  private static final Duration RETENTION_PERIOD = Duration.ofDays(30);

  private final Context context = ApplicationProvider.getApplicationContext();
  private final FakeTimeSource timeSource = new FakeTimeSource();
  private final ListeningExecutorService executorService =
      MoreExecutors.listeningDecorator(Executors.newSingleThreadExecutor());
  private TaskContributionTracker tracker;

  @Mock private StabilityMonitor mockStabilityMonitor;

  @Before
  public void setUp() throws Exception {
    tracker =
        TaskContributionTracker.create(
            context, timeSource, executorService, RETENTION_PERIOD, mockStabilityMonitor);
  }

  @Test
  public void commitContributions_newTask_success() throws Exception {
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1)).get();
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1);
  }

  @Test
  public void commitContributions_existingTask_success() throws Exception {
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1)).get();
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1);

    timeSource.advance(Duration.ofSeconds(1));
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_2)).get();
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1, ENTRY_ID_2);
  }

  @Test
  public void commitContributions_multipleEntries_success() throws Exception {
    tracker
        .commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1, ENTRY_ID_2))
        .get();
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1, ENTRY_ID_2);
  }

  @Test
  public void commitContributions_multipleTasks_success() throws Exception {
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1)).get();
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_2, ImmutableList.of(ENTRY_ID_2)).get();
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1);
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_2).get())
        .containsExactly(ENTRY_ID_2);
  }

  @Test
  public void rollbackContributions_removesFromDb() throws Exception {
    tracker
        .commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1, ENTRY_ID_2))
        .get();
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1, ENTRY_ID_2);

    tracker.rollbackContributions(LOG_SOURCE_1, TASK_NAME_1).get();
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get()).isEmpty();
  }

  @Test
  public void rollbackContributions_multipleTasks_onlyRemovesOne() throws Exception {
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1)).get();
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_2, ImmutableList.of(ENTRY_ID_2)).get();

    tracker.rollbackContributions(LOG_SOURCE_1, TASK_NAME_1).get();
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get()).isEmpty();
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_2).get())
        .containsExactly(ENTRY_ID_2);
  }

  @Test
  public void getContributedIds_differentTask_returnsEmpty() throws Exception {
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1)).get();
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_2).get()).isEmpty();
  }

  @Test
  public void getContributedIds_differentPrivateDataSource_returnsEmpty() throws Exception {
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1)).get();
    assertThat(tracker.getContributedIds(LOG_SOURCE_2, TASK_NAME_1).get()).isEmpty();
  }

  @Test
  public void commitContributions_sameTaskNameDifferentPrivateDataSources_success()
      throws Exception {
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1)).get();
    tracker.commitContributions(LOG_SOURCE_2, TASK_NAME_1, ImmutableList.of(ENTRY_ID_2)).get();

    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1);
    assertThat(tracker.getContributedIds(LOG_SOURCE_2, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_2);
  }

  @Test
  public void rollbackContributions_sameTaskNameDifferentPrivateDataSources_onlyRemovesOne()
      throws Exception {
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1)).get();
    tracker.commitContributions(LOG_SOURCE_2, TASK_NAME_1, ImmutableList.of(ENTRY_ID_2)).get();

    tracker.rollbackContributions(LOG_SOURCE_1, TASK_NAME_1).get();

    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get()).isEmpty();
    assertThat(tracker.getContributedIds(LOG_SOURCE_2, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_2);
  }

  @Test
  public void rollbackContributions_noContributionsForPrivateDataSource_doesNothing()
      throws Exception {
    // Commit contribution for LOG_SOURCE_1
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1)).get();
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1);

    // Rollback for LOG_SOURCE_2, which has no contributions.
    tracker.rollbackContributions(LOG_SOURCE_2, TASK_NAME_1).get();

    // Verify LOG_SOURCE_1 is unaffected.
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1);
    // Verify LOG_SOURCE_2 has no contributions.
    assertThat(tracker.getContributedIds(LOG_SOURCE_2, TASK_NAME_1).get()).isEmpty();
  }

  @Test
  public void rollbackContributions_nullLogSource_doesNothing() throws Exception {
    // Commit contribution for LOG_SOURCE_1
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1)).get();
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1);

    // Rollback for null log source, which has no contributions.
    // If rollbackContributions failed, get() would throw an exception.
    tracker.rollbackContributions(null, TASK_NAME_1).get();

    // Verify LOG_SOURCE_1 is unaffected.
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1);
  }

  @Test
  public void commitContributions_nullTaskName_doesNothing() throws Exception {
    // Commit contribution for TASK_NAME_1
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1)).get();
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1);

    // Commit for null task name.
    // If commitContributions failed, get() would throw an exception.
    tracker.commitContributions(LOG_SOURCE_1, null, ImmutableList.of(ENTRY_ID_2)).get();

    // Verify TASK_NAME_1 is unaffected.
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1);
    // Verify null task name has no contributions.
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, null).get()).isEmpty();
  }

  @Test
  public void rollbackContributions_nullTaskName_doesNothing() throws Exception {
    // Commit contribution for TASK_NAME_1
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1)).get();
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1);

    // Rollback for null task name, which has no contributions.
    // If rollbackContributions failed, get() would throw an exception.
    tracker.rollbackContributions(LOG_SOURCE_1, null).get();

    // Verify TASK_NAME_1 is unaffected.
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1);
  }

  @Test
  public void cleanupOldEntries_removesOld() throws Exception {
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1)).get();
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_2, ImmutableList.of(ENTRY_ID_2)).get();

    timeSource.advance(RETENTION_PERIOD.minusDays(15));
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_3)).get();
    // Advance so ENTRY_ID_1 and ENTRY_ID_2 are old, but ENTRY_ID_3 is not.
    timeSource.advance(Duration.ofDays(16));

    // Create a new tracker to trigger cleanup on creation.
    TaskContributionTracker newTracker =
        TaskContributionTracker.create(
            context, timeSource, executorService, RETENTION_PERIOD, mockStabilityMonitor);
    assertThat(newTracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_3); // New
    assertThat(newTracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_2).get()).isEmpty(); // Old
  }

  @Test
  public void cleanupOldEntries_removesEmptyTasks() throws Exception {
    assertThat(tracker.getTaskCount().get()).isEqualTo(0);
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1)).get();
    assertThat(tracker.getTaskCount().get()).isEqualTo(1);

    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_2, ImmutableList.of(ENTRY_ID_2)).get();
    assertThat(tracker.getTaskCount().get()).isEqualTo(2);
    timeSource.advance(RETENTION_PERIOD.plusDays(1));

    // Create a new tracker to trigger cleanup on creation.
    TaskContributionTracker newTracker =
        TaskContributionTracker.create(
            context, timeSource, executorService, RETENTION_PERIOD, mockStabilityMonitor);
    assertThat(newTracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get()).isEmpty();
    // Assert that the empty tasks are removed.
    assertThat(newTracker.getTaskCount().get()).isEqualTo(0);

    // Check that entries are gone for the other task as well.
    assertThat(newTracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_2).get()).isEmpty();
    assertThat(newTracker.getTaskCount().get()).isEqualTo(0);

    // Add a new contribution to the same task name, it should be treated as a new task.
    timeSource.advance(Duration.ofDays(1));
    newTracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_3)).get();
    assertThat(newTracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_3);
    assertThat(newTracker.getTaskCount().get()).isEqualTo(1);
  }

  @Test
  public void cleanupOldEntries_keepsNonEmptyTasks() throws Exception {
    tracker
        .commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1))
        .get(); // Will become old
    timeSource.advance(RETENTION_PERIOD.minusDays(15));
    tracker
        .commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_2))
        .get(); // Will be new
    timeSource.advance(Duration.ofDays(16));

    // Create a new tracker to trigger cleanup on creation.
    TaskContributionTracker newTracker =
        TaskContributionTracker.create(
            context, timeSource, executorService, RETENTION_PERIOD, mockStabilityMonitor);
    assertThat(newTracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_2); // New
  }

  @Test
  public void getContributedIds_noTask_returnsEmpty() throws Exception {
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get()).isEmpty();
  }

  @Test
  public void getContributedIds_singleContribution_returnsId() throws Exception {
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1)).get();
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1);
  }

  @Test
  public void getContributedIds_multipleContributions_returnsIds() throws Exception {
    tracker
        .commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1, ENTRY_ID_2))
        .get();
    timeSource.advance(Duration.ofSeconds(1));
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_3)).get();
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1, ENTRY_ID_2, ENTRY_ID_3);
  }

  @Test
  public void getContributedIds_multipleTasks_returnsIdsForCorrectTask() throws Exception {
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1)).get();
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_2, ImmutableList.of(ENTRY_ID_2)).get();
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1);
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_2).get())
        .containsExactly(ENTRY_ID_2);
  }

  @Test
  public void clearCommittedContributions_preventsRollback() throws Exception {
    // Commit some contributions.
    tracker
        .commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1, ENTRY_ID_2))
        .get();
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1, ENTRY_ID_2);

    // Clear the in-memory map.
    tracker.clearCommittedContributions();

    // Rollback should now be a no-op because the in-memory tracking was cleared.
    tracker.rollbackContributions(LOG_SOURCE_1, TASK_NAME_1).get();

    // The DB records should still be intact since clearCommittedContributions only clears the
    // in-memory map, not the database.
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1, ENTRY_ID_2);
  }

  @Test
  public void clearCommittedContributions_clearsAllLogSources() throws Exception {
    // Commit contributions for two different log sources.
    tracker.commitContributions(LOG_SOURCE_1, TASK_NAME_1, ImmutableList.of(ENTRY_ID_1)).get();
    tracker.commitContributions(LOG_SOURCE_2, TASK_NAME_1, ImmutableList.of(ENTRY_ID_2)).get();

    // Clear the in-memory map.
    tracker.clearCommittedContributions();

    // Rollback should be a no-op for both log sources.
    tracker.rollbackContributions(LOG_SOURCE_1, TASK_NAME_1).get();
    tracker.rollbackContributions(LOG_SOURCE_2, TASK_NAME_1).get();

    // DB records should still be intact for both.
    assertThat(tracker.getContributedIds(LOG_SOURCE_1, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_1);
    assertThat(tracker.getContributedIds(LOG_SOURCE_2, TASK_NAME_1).get())
        .containsExactly(ENTRY_ID_2);
  }
}
