// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __THREAD_POOL_H__
#define __THREAD_POOL_H__

#include <cstdlib>
#include <utility>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <string>


class ThreadPool {
 public:
  // Basic unit of work that our threads do
  typedef std::function<void(int)> Work;

  ThreadPool(int num_thread, int device_id, bool set_affinity);

  ~ThreadPool();

  /**
   * @brief Adds work to the queue with optional priority.
   *        The work only gets queued and it will only start after invoking
   *        `RunAll` (wakes up all threads to complete all remaining works) or
   *        `DoWorkWithID` (wakes up a single thread to complete one work unit).
   * @remarks if finished_adding_work == true, the thread pool will proceed picking
   *          tasks from its queue, otherwise it will hold execution until `RunAll`
   *          is invoked.
   */
  void AddWork(Work work, int64_t priority = 0, bool finished_adding_work = false);

  /**
   * @brief Adds work to the queue with optional priority and wakes up a single
   *        thread that will pick the task in the queue with highest priority.
   */
  void DoWorkWithID(Work work, int64_t priority = 0);

  /**
   * @brief Wakes up all the threads to complete all the queued work,
   *        optionally not waiting for the work to be finished before return
   *        (the default wait=true is equivalent to invoking WaitForWork after RunAll).
   */
  void RunAll(bool wait = true);

  /**
   * @brief Waits until all work issued to the thread pool is complete
   */
  void WaitForWork(bool checkForErrors = true);

  int size() const;

  std::vector<std::thread::id> GetThreadIds() const;

  // delete certain operators
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

 private:
  void ThreadMain(int thread_id, int device_id, bool set_affinity);

  std::vector<std::thread> threads_;

  using PrioritizedWork = std::pair<int64_t, Work>;
  struct SortByPriority {
    bool operator() (const PrioritizedWork &a, const PrioritizedWork &b) {
      return a.first < b.first;
    }
  };
  std::priority_queue<PrioritizedWork, std::vector<PrioritizedWork>, SortByPriority> work_queue_;

  bool running_;
  bool work_complete_;
  bool adding_work_;
  int active_threads_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::condition_variable completed_;

  //  Stored error strings for each thread
  std::vector<std::queue<std::string>> tl_errors_;
};


#endif  // __THREAD_POOL_H__
