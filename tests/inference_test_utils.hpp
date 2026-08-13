/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace nve_test {

struct ProcessResult {
  int exit_code;
  std::string output;
};

inline ProcessResult run_process(
    const std::vector<std::string>& arguments,
    std::chrono::seconds timeout = std::chrono::seconds{60}) {
  if (arguments.empty()) {
    throw std::invalid_argument("Cannot run a process without an executable");
  }
  if (timeout <= std::chrono::seconds::zero()) {
    throw std::invalid_argument("Process timeout must be greater than zero");
  }

  std::array<int, 2> output_pipe{};
  if (pipe(output_pipe.data()) != 0) {
    throw std::runtime_error("pipe failed: " + std::string(std::strerror(errno)));
  }

  const pid_t child = fork();
  if (child < 0) {
    close(output_pipe[0]);
    close(output_pipe[1]);
    throw std::runtime_error("fork failed: " + std::string(std::strerror(errno)));
  }

  if (child == 0) {
    close(output_pipe[0]);
    if (dup2(output_pipe[1], STDOUT_FILENO) < 0 ||
        dup2(output_pipe[1], STDERR_FILENO) < 0) {
      _exit(126);
    }
    close(output_pipe[1]);

    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (const std::string& argument : arguments) {
      argv.push_back(const_cast<char*>(argument.c_str()));
    }
    argv.push_back(nullptr);
    execv(argv[0], argv.data());
    _exit(127);
  }

  close(output_pipe[1]);
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::string output;
  std::array<char, 4096> buffer{};
  int status = 0;
  bool child_exited = false;
  bool pipe_open = true;
  while (!child_exited || pipe_open) {
    if (!child_exited) {
      const pid_t wait_result = waitpid(child, &status, WNOHANG);
      if (wait_result == child) {
        child_exited = true;
      } else if (wait_result < 0 && errno != EINTR) {
        close(output_pipe[0]);
        throw std::runtime_error("waitpid failed: " +
                                 std::string(std::strerror(errno)));
      }

      if (!child_exited && std::chrono::steady_clock::now() >= deadline) {
        const int kill_result = kill(child, SIGKILL);
        const int kill_error = errno;
        if (pipe_open) {
          close(output_pipe[0]);
          pipe_open = false;
        }

        while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
        }

        std::string message = "Process timed out after " +
                              std::to_string(timeout.count()) + " seconds";
        if (kill_result != 0 && kill_error != ESRCH) {
          message += "; kill failed: " + std::string(std::strerror(kill_error));
        }
        if (!output.empty()) {
          message += "\nProcess output:\n" + output;
        }
        throw std::runtime_error(message);
      }
    }

    int poll_timeout_ms = 100;
    const auto remaining = deadline - std::chrono::steady_clock::now();
    const auto remaining_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(remaining);
    if (!child_exited &&
        remaining_ms < std::chrono::milliseconds(poll_timeout_ms)) {
      poll_timeout_ms = static_cast<int>(remaining_ms.count());
      if (poll_timeout_ms < 1) {
        poll_timeout_ms = 1;
      }
    }

    pollfd descriptor{output_pipe[0], POLLIN, 0};
    const int poll_result = poll(pipe_open ? &descriptor : nullptr,
                                 pipe_open ? 1 : 0, poll_timeout_ms);
    if (poll_result < 0) {
      if (errno == EINTR) {
        continue;
      }
      const int poll_error = errno;
      close(output_pipe[0]);
      if (!child_exited) {
        kill(child, SIGKILL);
        while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
        }
      }
      throw std::runtime_error("poll failed: " +
                               std::string(std::strerror(poll_error)));
    }
    if (poll_result == 0) {
      continue;
    }
    if ((descriptor.revents & POLLNVAL) != 0) {
      close(output_pipe[0]);
      if (!child_exited) {
        kill(child, SIGKILL);
        while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
        }
      }
      throw std::runtime_error("poll failed: invalid output pipe");
    }

    const ssize_t bytes_read = read(output_pipe[0], buffer.data(), buffer.size());
    if (bytes_read > 0) {
      output.append(buffer.data(), static_cast<std::size_t>(bytes_read));
    } else if (bytes_read == 0) {
      close(output_pipe[0]);
      pipe_open = false;
    } else if (errno != EINTR) {
      const int read_error = errno;
      close(output_pipe[0]);
      if (!child_exited) {
        kill(child, SIGKILL);
        while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
        }
      }
      throw std::runtime_error("read failed: " +
                               std::string(std::strerror(read_error)));
    }
  }

  if (WIFEXITED(status)) {
    return {WEXITSTATUS(status), std::move(output)};
  }
  if (WIFSIGNALED(status)) {
    return {128 + WTERMSIG(status), std::move(output)};
  }
  return {-1, std::move(output)};
}

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(const std::string& prefix) {
    std::string pattern =
        (std::filesystem::temp_directory_path() / (prefix + "-XXXXXX")).string();
    std::vector<char> writable_pattern(pattern.begin(), pattern.end());
    writable_pattern.push_back('\0');
    char* directory = mkdtemp(writable_pattern.data());
    if (directory == nullptr) {
      throw std::runtime_error("mkdtemp failed: " + std::string(std::strerror(errno)));
    }
    path_ = directory;
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

}  // namespace nve_test
