/*
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fcp/client/cache/file_backed_resource_cache.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>  // NOLINT
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "google/protobuf/timestamp.pb.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/platform.h"
#include "fcp/base/time_util.h"
#include "fcp/client/cache/cache_manifest.pb.h"
#include "fcp/client/diag_codes.pb.h"
#include "protostore/file-storage.h"
#include "protostore/proto-data-store.h"

namespace fcp {
namespace client {
namespace cache {

constexpr absl::string_view kCacheManifestFileName = "cache_manifest.pb";
constexpr absl::string_view kParentDir = "fcp";
// Cached files will be saved in <cache directory>/fcp/cache.
constexpr absl::string_view kCacheDir = "cache";

absl::StatusOr<CacheManifest> FileBackedResourceCache::ReadInternal() {
  absl::StatusOr<const CacheManifest*> data = pds_->Read();
  if (data.ok()) {
    return *data.value();
  }
  log_manager_.LogDiag(ProdDiagCode::RESOURCE_CACHE_MANIFEST_READ_FAILED);
  return absl::InternalError(
      absl::StrCat("Failed to read from database, with error message: ",
                   data.status().message()));
}

absl::Status FileBackedResourceCache::WriteInternal(
    std::unique_ptr<CacheManifest> manifest) {
  absl::Status status = pds_->Write(std::move(manifest));
  if (!status.ok()) {
    log_manager_.LogDiag(ProdDiagCode::RESOURCE_CACHE_MANIFEST_WRITE_FAILED);
  }
  return status;
}

absl::StatusOr<std::unique_ptr<FileBackedResourceCache>>
FileBackedResourceCache::Create(absl::string_view base_dir,
                                absl::string_view cache_dir,
                                LogManager* log_manager, fcp::Clock* clock,
                                int64_t max_cache_size_bytes) {
  // Create <cache root>/fcp.
  std::filesystem::path cache_root_path(cache_dir);
  if (!cache_root_path.is_absolute()) {
    return absl::InvalidArgumentError(
        absl::StrCat("The provided path: ", cache_dir,
                     "is invalid. The path must be absolute"));
  }
  std::filesystem::path cache_dir_path =
      cache_root_path / kParentDir / kCacheDir;
  std::error_code error;
  std::filesystem::create_directories(cache_dir_path, error);
  if (error.value() != 0) {
    return absl::InternalError(absl::StrCat(
        "Failed to create FileBackedResourceCache cache directory ",
        cache_dir_path.string()));
  }
  // Create <files root>/fcp/cache_manifest.pb.
  std::filesystem::path manifest_path(base_dir);
  if (!manifest_path.is_absolute()) {
    return absl::InvalidArgumentError(
        absl::StrCat("The provided path: ", manifest_path.string(),
                     "is invalid. The path must start with \"/\""));
  }
  manifest_path /= kParentDir;
  std::filesystem::create_directories(manifest_path, error);
  if (error.value() != 0) {
    return absl::InternalError(
        absl::StrCat("Failed to create directory ", manifest_path.string()));
  }
  manifest_path /= kCacheManifestFileName;

  auto file_storage = std::make_unique<protostore::FileStorage>();
  auto pds = std::make_unique<protostore::ProtoDataStore<CacheManifest>>(
      *file_storage, manifest_path.string());
  std::unique_ptr<FileBackedResourceCache> resource_cache =
      absl::WrapUnique(new FileBackedResourceCache(
          std::move(pds), std::move(file_storage), cache_dir_path,
          manifest_path, log_manager, clock, max_cache_size_bytes));

  FCP_RETURN_IF_ERROR(resource_cache->Initialize());

  return resource_cache;
}

absl::Status FileBackedResourceCache::Put(absl::string_view cache_id,
                                          const absl::Cord& resource,
                                          const google::protobuf::Any& metadata,
                                          absl::Duration max_age) {
  absl::MutexLock lock(&mutex_);

  if (resource.size() > max_cache_size_bytes_ / 2) {
    return absl::ResourceExhaustedError(absl::StrCat(cache_id, " too large"));
  }

  FCP_ASSIGN_OR_RETURN(CacheManifest manifest, ReadInternal());

  FCP_RETURN_IF_ERROR(CleanUp(resource.size(), manifest));

  std::filesystem::path cached_file_path = cache_dir_path_ / cache_id;
  absl::Time now = clock_.Now();
  absl::Time expiry = now + max_age;
  CachedResource cached_resource;
  cached_resource.set_file_name(std::string(cache_id));
  *cached_resource.mutable_metadata() = metadata;
  *cached_resource.mutable_expiry_time() =
      TimeUtil::ConvertAbslToProtoTimestamp(expiry);
  *cached_resource.mutable_last_accessed_time() =
      TimeUtil::ConvertAbslToProtoTimestamp(now);

  // Write the manifest back to disk before we write the file.
  manifest.mutable_cache()->insert({std::string(cache_id), cached_resource});
  absl::Status status =
      WriteInternal(std::make_unique<CacheManifest>(std::move(manifest)));

  // Write file if it doesn't exist.
  if (!std::filesystem::exists(cached_file_path)) {
    auto status = WriteCordToFile(cached_file_path.string(), resource);
    if (!status.ok()) {
      log_manager_.LogDiag(ProdDiagCode::RESOURCE_CACHE_RESOURCE_WRITE_FAILED);
      return status;
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<FileBackedResourceCache::ResourceAndMetadata>
FileBackedResourceCache::Get(absl::string_view cache_id,
                             std::optional<absl::Duration> max_age) {
  absl::MutexLock lock(&mutex_);
  FCP_ASSIGN_OR_RETURN(CacheManifest manifest, ReadInternal());

  if (!manifest.cache().contains(cache_id)) {
    return absl::NotFoundError(absl::StrCat(cache_id, " not found"));
  }
  CachedResource cached_resource = manifest.cache().at(cache_id);
  std::filesystem::path cached_file_path = cache_dir_path_ / cache_id;
  google::protobuf::Any metadata = cached_resource.metadata();
  absl::Time now = clock_.Now();
  *cached_resource.mutable_last_accessed_time() =
      TimeUtil::ConvertAbslToProtoTimestamp(now);
  if (max_age.has_value()) {
    absl::Time expiry = now + max_age.value();
    *cached_resource.mutable_expiry_time() =
        TimeUtil::ConvertAbslToProtoTimestamp(expiry);
  }

  absl::StatusOr<absl::Cord> contents =
      ReadFileToCord(cached_file_path.string());
  if (!contents.ok()) {
    log_manager_.LogDiag(ProdDiagCode::RESOURCE_CACHE_RESOURCE_READ_FAILED);
    manifest.mutable_cache()->erase(cache_id);
    std::error_code error;
    std::filesystem::remove(cached_file_path, error);
    if (error.value() != 0) {
      return absl::InternalError(error.message());
    }
    // Treat as not found, the resource should be fetched again.
    return absl::NotFoundError(absl::StrCat(cache_id, " not found"));
  }

  std::string cache_id_string(cache_id);
  manifest.mutable_cache()->erase(cache_id_string);
  manifest.mutable_cache()->insert({cache_id_string, cached_resource});

  absl::Status status =
      WriteInternal(std::make_unique<CacheManifest>(std::move(manifest)));
  if (!status.ok()) return status;

  return FileBackedResourceCache::ResourceAndMetadata{*contents, metadata};
}

absl::Status FileBackedResourceCache::Initialize() {
  absl::MutexLock lock(&mutex_);
  std::string pds_path = manifest_path_.string();
  if (!std::filesystem::exists(pds_path)) {
    std::ofstream ofs(pds_path);
  }
  FCP_ASSIGN_OR_RETURN(absl::StatusOr<int64_t> file_size,
                       storage_->GetFileSize(pds_path));
  // Initialize db if it's not initialized.
  if (file_size.value() == 0) {
    FCP_RETURN_IF_ERROR(WriteInternal(std::make_unique<CacheManifest>()));
  }
  // Then run CleanUp. Even if our manifest was empty we still might have
  // stranded cache files to delete, i.e. in the case that the manifest was
  // deleted but the cache dir was not deleted.
  FCP_ASSIGN_OR_RETURN(CacheManifest manifest, ReadInternal());
  FCP_RETURN_IF_ERROR(CleanUp(std::nullopt, manifest));
  FCP_RETURN_IF_ERROR(pds_->Write(std::make_unique<CacheManifest>(manifest)));

  return absl::OkStatus();
}

absl::Status FileBackedResourceCache::CleanUp(
    std::optional<int64_t> reserved_space_bytes, CacheManifest& manifest) {
  // Expire any cached resources past their expiry.
  // Clean up any files that are not tracked in the manifest.
  // Clean up any manifest entries that point to nonexistent files.

  // In order to delete files we don't track in the CacheManifest (or that
  // became untracked due to a crash), fill files_to_delete with every file in
  // the cache dir. We'll then remove any file not actively in the cache.
  std::set<std::filesystem::path> files_to_delete;

  // We don't have any subdirectories in the cache, so we can use a directory
  // iterator.
  std::error_code directory_error;
  for (auto& file :
       std::filesystem::directory_iterator(cache_dir_path_, directory_error)) {
    files_to_delete.insert(cache_dir_path_ / file);
  }

  if (directory_error.value() != 0) {
    return absl::InternalError(absl::StrCat("Error iterating over cache dir: ",
                                            directory_error.message()));
  }

  int64_t max_allowed_size_bytes = max_cache_size_bytes_;
  max_allowed_size_bytes -= reserved_space_bytes.value_or(0);

  std::set<std::string> cache_ids_to_delete;
  absl::Time now = clock_.Now();
  for (const auto& entry : manifest.cache()) {
    CachedResource cached_resource = entry.second;
    absl::Time expiry =
        TimeUtil::ConvertProtoToAbslTime(cached_resource.expiry_time());
    if (expiry < now) {
      cache_ids_to_delete.insert(entry.first);
    } else {
      std::filesystem::path actively_cached_file =
          cache_dir_path_ / entry.second.file_name();
      files_to_delete.erase(actively_cached_file);
    }
  }

  // Then delete CacheManifest entries.
  for (const auto& cache_id : cache_ids_to_delete) {
    manifest.mutable_cache()->erase(cache_id);
  }

  // Then delete files.
  absl::Status filesystem_status = absl::OkStatus();
  for (const auto& file : files_to_delete) {
    std::error_code remove_error;
    std::filesystem::remove(file, remove_error);
    // We intentionally loop through all files and attempt to remove as many as
    // we can, then return the last error we saw.
    if (remove_error.value() != 0) {
      filesystem_status = absl::InternalError(
          absl::StrCat("Failed to delete file: ", remove_error.message()));
    }
  }

  FCP_RETURN_IF_ERROR(filesystem_status);

  // If we still exceed the allowed size of the cache, delete entries until
  // we're under the allowed size, sorted by least recently used.

  // Build up a list of (cache_id, least recently used timestamp) and compute
  // the total size of the cache.
  std::vector<std::pair<std::string, absl::Time>> cache_id_lru;
  cache_id_lru.reserve(manifest.cache().size());
  uintmax_t cache_dir_size = 0;

  for (auto& [id, resource] : manifest.cache()) {
    std::error_code file_size_error;
    cache_id_lru.emplace_back(std::make_pair(
        id, TimeUtil::ConvertProtoToAbslTime(resource.last_accessed_time())));
    // We calculate the sum of tracked files instead of taking the file_size()
    // of the cache directory, because the latter generally does not reflect the
    // the total size of all of the files inside a directory.
    cache_dir_size += std::filesystem::file_size(
        cache_dir_path_ / resource.file_name(), file_size_error);
    // Loop through as many as we can and if there's an error, return the most
    // recent one.
    if (file_size_error.value() != 0) {
      filesystem_status = absl::InternalError(
          absl::StrCat("Error getting file size: ", file_size_error.message()));
    }
  }

  FCP_RETURN_IF_ERROR(filesystem_status);

  // Then, if the cache is bigger than the allowed size, delete entries ordered
  // by least recently used until we're below the threshold.
  if (cache_dir_size > max_allowed_size_bytes) {
    std::sort(cache_id_lru.begin(), cache_id_lru.end(),
              [](std::pair<std::string, absl::Time> first,
                 std::pair<std::string, absl::Time> second) -> bool {
                // Sort by least recently used timestamp.
                return first.second < second.second;
              });
    for (auto const& entry : cache_id_lru) {
      std::string id_to_remove = entry.first;
      std::filesystem::path file_to_remove =
          cache_dir_path_ / manifest.cache().at(id_to_remove).file_name();
      manifest.mutable_cache()->erase(id_to_remove);
      std::error_code remove_error;
      uintmax_t file_size =
          std::filesystem::file_size(file_to_remove, remove_error);
      std::filesystem::remove(file_to_remove, remove_error);
      if (remove_error.value() != 0 && filesystem_status.ok()) {
        filesystem_status = absl::InternalError(
            absl::StrCat("Failed to delete file: ", remove_error.message()));
      }
      cache_dir_size -= file_size;
      if (cache_dir_size < max_allowed_size_bytes) break;
    }
  }

  FCP_RETURN_IF_ERROR(filesystem_status);

  return absl::OkStatus();
}

}  // namespace cache
}  // namespace client
}  // namespace fcp
