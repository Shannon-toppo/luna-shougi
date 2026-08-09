#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace luna::match {

// A child process with its standard input and output connected to pipes.
//
// Whatever the child writes is drained by a background thread, so an engine
// streaming "info" lines while nobody happens to be reading can never fill
// the pipe buffer and block. The child's standard error goes down the same
// pipe: a crashing engine says something useful there, and a caller reading
// USI throws away lines it does not recognise anyway.
class Process {
 public:
  Process();
  ~Process();
  Process(const Process&) = delete;
  Process& operator=(const Process&) = delete;

  // `args[0]` is the executable to run; the child's working directory is set
  // to the directory it lives in, which is where engines look for their own
  // files. Returns false and fills `error` when the child could not start.
  bool Start(const std::vector<std::string>& args, std::string& error);
  bool Started() const;

  // False once the pipe is closed, which means the child is gone.
  bool WriteLine(const std::string& line);

  // Waits up to `timeout` for one line. False on timeout, or once the child's
  // output has ended and every line already read has been handed back.
  bool ReadLine(std::string& line, std::chrono::milliseconds timeout);

  // Closes the child's input so it can leave on its own, waits `grace`, then
  // kills it. Safe to call more than once; the destructor calls it.
  void Stop(std::chrono::milliseconds grace = std::chrono::milliseconds(2000));

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace luna::match
