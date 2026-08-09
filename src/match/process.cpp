#include "match/process.hpp"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstring>
#endif

namespace luna::match {
namespace {

// The directory `path` lives in, or "" when it has no directory part.
std::string ParentDirectory(const std::string& path) {
  const size_t at = path.find_last_of("/\\");
  return at == std::string::npos ? std::string() : path.substr(0, at);
}

#ifdef _WIN32

std::string Narrow(const std::wstring& wide) {
  if (wide.empty()) return {};
  const int size = WideCharToMultiByte(
      CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), nullptr, 0, nullptr, nullptr);
  std::string utf8(static_cast<size_t>(size), '\0');
  WideCharToMultiByte(
      CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), utf8.data(), size, nullptr, nullptr);
  return utf8;
}

std::wstring Widen(const std::string& utf8) {
  if (utf8.empty()) return {};
  const int size =
      MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
  std::wstring wide(static_cast<size_t>(size), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), wide.data(), size);
  return wide;
}

// Windows hands the child one string and lets its runtime split it again, so
// an argument has to be quoted the way that splitter expects: backslashes
// only double up when they are what a quote is standing behind.
std::string QuoteArgument(const std::string& arg) {
  if (!arg.empty() && arg.find_first_of(" \t\"") == std::string::npos) return arg;

  std::string quoted = "\"";
  size_t backslashes = 0;
  for (const char c : arg) {
    if (c == '\\') {
      ++backslashes;
      continue;
    }
    if (c == '"') {
      quoted.append(backslashes * 2 + 1, '\\');
      backslashes = 0;
      quoted += '"';
      continue;
    }
    quoted.append(backslashes, '\\');
    backslashes = 0;
    quoted += c;
  }
  quoted.append(backslashes * 2, '\\');
  quoted += '"';
  return quoted;
}

// The child is started in its own directory, so a path relative to ours
// would be measured from the wrong place once it is running. Resolving it
// here also means the error message names the file actually looked for.
std::string AbsolutePath(const std::string& path) {
  const std::wstring wide = Widen(path);
  const DWORD needed = GetFullPathNameW(wide.c_str(), 0, nullptr, nullptr);
  if (needed == 0) return path;
  std::wstring full(needed, L'\0');
  const DWORD length = GetFullPathNameW(wide.c_str(), needed, full.data(), nullptr);
  if (length == 0 || length >= needed) return path;
  full.resize(length);
  return Narrow(full);
}

std::string LastErrorMessage() {
  const DWORD code = GetLastError();
  char* text = nullptr;
  const DWORD length = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr,
      code,
      0,
      reinterpret_cast<char*>(&text),
      0,
      nullptr);
  std::string message = length > 0 ? std::string(text, length) : "error " + std::to_string(code);
  if (text != nullptr) LocalFree(text);
  while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) {
    message.pop_back();
  }
  return message;
}

#else  // POSIX

// As above. A bare name with no directory in it is left alone: that is a
// request to search PATH, which execvp does and this cannot.
std::string AbsolutePath(const std::string& path) {
  if (path.find('/') == std::string::npos) return path;
  if (!path.empty() && path.front() == '/') return path;
  std::string cwd(4096, '\0');
  if (getcwd(cwd.data(), cwd.size()) == nullptr) return path;
  cwd.resize(std::strlen(cwd.c_str()));
  return cwd + '/' + path;
}

#endif  // _WIN32

}  // namespace

struct Process::Impl {
  std::mutex mutex;
  std::condition_variable ready;
  std::deque<std::string> lines;
  bool eof = false;
  bool started = false;
  std::thread reader;

#ifdef _WIN32
  HANDLE process = nullptr;
  HANDLE main_thread = nullptr;
  HANDLE stdin_write = nullptr;
  HANDLE stdout_read = nullptr;
#else
  pid_t pid = -1;
  int stdin_write = -1;
  int stdout_read = -1;
#endif

  // Bytes from the child, or <= 0 once its output has ended.
  int ReadSome(char* buffer, int size);
  bool WriteAll(const char* data, size_t size);
  void CloseInput();
  void CloseOutput();
  void Kill();
  bool WaitFor(std::chrono::milliseconds timeout);
};

#ifdef _WIN32

int Process::Impl::ReadSome(char* buffer, int size) {
  if (stdout_read == nullptr) return 0;
  DWORD got = 0;
  if (!ReadFile(stdout_read, buffer, static_cast<DWORD>(size), &got, nullptr)) return 0;
  return static_cast<int>(got);
}

bool Process::Impl::WriteAll(const char* data, size_t size) {
  if (stdin_write == nullptr) return false;
  while (size > 0) {
    DWORD written = 0;
    if (!WriteFile(stdin_write, data, static_cast<DWORD>(size), &written, nullptr) ||
        written == 0) {
      return false;
    }
    data += written;
    size -= written;
  }
  return true;
}

void Process::Impl::CloseInput() {
  if (stdin_write != nullptr) {
    CloseHandle(stdin_write);
    stdin_write = nullptr;
  }
}

void Process::Impl::CloseOutput() {
  if (stdout_read != nullptr) {
    CloseHandle(stdout_read);
    stdout_read = nullptr;
  }
}

void Process::Impl::Kill() {
  if (process == nullptr) return;
  TerminateProcess(process, 1);
  WaitForSingleObject(process, INFINITE);
}

bool Process::Impl::WaitFor(std::chrono::milliseconds timeout) {
  if (process == nullptr) return true;
  return WaitForSingleObject(process, static_cast<DWORD>(timeout.count())) == WAIT_OBJECT_0;
}

#else  // POSIX

int Process::Impl::ReadSome(char* buffer, int size) {
  if (stdout_read < 0) return 0;
  ssize_t got = 0;
  do {
    got = read(stdout_read, buffer, static_cast<size_t>(size));
  } while (got < 0 && errno == EINTR);
  return static_cast<int>(got);
}

bool Process::Impl::WriteAll(const char* data, size_t size) {
  if (stdin_write < 0) return false;
  while (size > 0) {
    const ssize_t written = write(stdin_write, data, size);
    if (written < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    data += written;
    size -= static_cast<size_t>(written);
  }
  return true;
}

void Process::Impl::CloseInput() {
  if (stdin_write >= 0) {
    close(stdin_write);
    stdin_write = -1;
  }
}

void Process::Impl::CloseOutput() {
  if (stdout_read >= 0) {
    close(stdout_read);
    stdout_read = -1;
  }
}

void Process::Impl::Kill() {
  if (pid <= 0) return;
  kill(pid, SIGKILL);
  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  pid = -1;
}

bool Process::Impl::WaitFor(std::chrono::milliseconds timeout) {
  if (pid <= 0) return true;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (true) {
    int status = 0;
    const pid_t done = waitpid(pid, &status, WNOHANG);
    if (done == pid) {
      pid = -1;
      return true;
    }
    if (done < 0 && errno != EINTR) {
      // Already reaped, or never ours. Either way there is nothing left to
      // wait for, and forgetting the id keeps a later kill from landing on
      // whatever process the system gives the number to next.
      pid = -1;
      return true;
    }
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

#endif

Process::Process() : impl_(std::make_unique<Impl>()) {}

Process::~Process() {
  Stop();
}

bool Process::Started() const {
  return impl_->started;
}

bool Process::Start(const std::vector<std::string>& args, std::string& error) {
  if (args.empty()) {
    error = "no executable given";
    return false;
  }
  if (impl_->started) {
    error = "already started";
    return false;
  }

  std::vector<std::string> resolved = args;
  resolved[0] = AbsolutePath(args[0]);
  const std::string directory = ParentDirectory(resolved[0]);

#ifdef _WIN32
  SECURITY_ATTRIBUTES inheritable{};
  inheritable.nLength = sizeof(inheritable);
  inheritable.bInheritHandle = TRUE;

  HANDLE child_stdin = nullptr;
  HANDLE child_stdout = nullptr;
  if (!CreatePipe(&child_stdin, &impl_->stdin_write, &inheritable, 0)) {
    error = "could not create the input pipe: " + LastErrorMessage();
    return false;
  }
  if (!CreatePipe(&impl_->stdout_read, &child_stdout, &inheritable, 0)) {
    error = "could not create the output pipe: " + LastErrorMessage();
    CloseHandle(child_stdin);
    impl_->CloseInput();
    return false;
  }
  // Only the child's ends of the two pipes may be inherited; if the parent's
  // ends were too, the read would never see end of file.
  SetHandleInformation(impl_->stdin_write, HANDLE_FLAG_INHERIT, 0);
  SetHandleInformation(impl_->stdout_read, HANDLE_FLAG_INHERIT, 0);

  std::string command;
  for (const std::string& arg : resolved) {
    if (!command.empty()) command += ' ';
    command += QuoteArgument(arg);
  }
  std::wstring command_line = Widen(command);
  const std::wstring working_dir = Widen(directory);

  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdInput = child_stdin;
  startup.hStdOutput = child_stdout;
  startup.hStdError = child_stdout;

  PROCESS_INFORMATION info{};
  const BOOL ok = CreateProcessW(nullptr,
                                 command_line.data(),
                                 nullptr,
                                 nullptr,
                                 TRUE,
                                 CREATE_NO_WINDOW,
                                 nullptr,
                                 working_dir.empty() ? nullptr : working_dir.c_str(),
                                 &startup,
                                 &info);
  const std::string failure = ok ? std::string() : LastErrorMessage();
  CloseHandle(child_stdin);
  CloseHandle(child_stdout);
  if (!ok) {
    error = "could not start " + resolved[0] + ": " + failure;
    impl_->CloseInput();
    impl_->CloseOutput();
    return false;
  }
  impl_->process = info.hProcess;
  impl_->main_thread = info.hThread;

#else
  int to_child[2] = {-1, -1};
  int from_child[2] = {-1, -1};
  if (pipe(to_child) != 0) {
    error = std::string("could not create the input pipe: ") + std::strerror(errno);
    return false;
  }
  if (pipe(from_child) != 0) {
    error = std::string("could not create the output pipe: ") + std::strerror(errno);
    close(to_child[0]);
    close(to_child[1]);
    return false;
  }

  std::vector<std::string> storage = resolved;
  std::vector<char*> argv;
  argv.reserve(storage.size() + 1);
  for (std::string& arg : storage) argv.push_back(arg.data());
  argv.push_back(nullptr);

  const pid_t pid = fork();
  if (pid < 0) {
    error = std::string("could not fork: ") + std::strerror(errno);
    close(to_child[0]);
    close(to_child[1]);
    close(from_child[0]);
    close(from_child[1]);
    return false;
  }
  if (pid == 0) {
    // Only async-signal-safe calls from here until execvp.
    dup2(to_child[0], STDIN_FILENO);
    dup2(from_child[1], STDOUT_FILENO);
    dup2(from_child[1], STDERR_FILENO);
    close(to_child[0]);
    close(to_child[1]);
    close(from_child[0]);
    close(from_child[1]);
    if (!directory.empty() && chdir(directory.c_str()) != 0) _exit(127);
    execvp(argv[0], argv.data());
    _exit(127);
  }

  close(to_child[0]);
  close(from_child[1]);
  impl_->pid = pid;
  impl_->stdin_write = to_child[1];
  impl_->stdout_read = from_child[0];
#endif

  impl_->started = true;
  Impl* impl = impl_.get();
  impl_->reader = std::thread([impl] {
    std::string pending;
    char buffer[4096];
    while (true) {
      const int got = impl->ReadSome(buffer, static_cast<int>(sizeof(buffer)));
      if (got <= 0) break;
      pending.append(buffer, static_cast<size_t>(got));

      size_t start = 0;
      while (true) {
        const size_t newline = pending.find('\n', start);
        if (newline == std::string::npos) break;
        std::string line = pending.substr(start, newline - start);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        {
          std::lock_guard<std::mutex> lock(impl->mutex);
          impl->lines.push_back(std::move(line));
        }
        impl->ready.notify_one();
        start = newline + 1;
      }
      pending.erase(0, start);
    }
    {
      std::lock_guard<std::mutex> lock(impl->mutex);
      impl->eof = true;
    }
    impl->ready.notify_all();
  });

  return true;
}

bool Process::WriteLine(const std::string& line) {
  if (!impl_->started) return false;
  const std::string with_newline = line + "\n";
  return impl_->WriteAll(with_newline.data(), with_newline.size());
}

bool Process::ReadLine(std::string& line, std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(impl_->mutex);
  if (!impl_->ready.wait_for(
          lock, timeout, [this] { return !impl_->lines.empty() || impl_->eof; })) {
    return false;
  }
  if (impl_->lines.empty())
    return false;  // end of output
  line = std::move(impl_->lines.front());
  impl_->lines.pop_front();
  return true;
}

void Process::Stop(std::chrono::milliseconds grace) {
  if (!impl_->started) return;
  impl_->started = false;

  // Closing the child's input is how a USI engine is told to leave; killing
  // it is the fallback for one that will not.
  impl_->CloseInput();
  if (!impl_->WaitFor(grace)) impl_->Kill();

  // The reader thread ends when the child's output pipe does, which cannot
  // happen before the child is gone. Only then is it safe to close our end.
  if (impl_->reader.joinable()) impl_->reader.join();
  impl_->CloseOutput();

#ifdef _WIN32
  if (impl_->process != nullptr) {
    CloseHandle(impl_->process);
    impl_->process = nullptr;
  }
  if (impl_->main_thread != nullptr) {
    CloseHandle(impl_->main_thread);
    impl_->main_thread = nullptr;
  }
#endif
}

}  // namespace luna::match
