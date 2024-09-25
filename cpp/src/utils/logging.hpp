#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <mutex>
#include <chrono>
#include <iomanip>

enum class LogLevel {
    INFO,
    WARNING,
    ERROR,
    DEBUG
};

// Logger class definition
class Logger {
public:
    // Singleton pattern to ensure only one instance of Logger
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void setLogFile(const std::string& filename) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (logFile_.is_open()) {
            logFile_.close();
        }
        logFile_.open(filename, std::ios::out | std::ios::app);
    }

    void setLogLevel(LogLevel level) {
        logLevel_ = level;
    }

    // Logging method that includes worker index, type, and level
    template<typename... Args>
    void log(int workerIndex, const std::string& workerType, LogLevel level, Args... args) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level >= logLevel_) {
            std::ostringstream oss;
            oss << getCurrentTime() << " [Worker " << workerIndex << " | " << workerType << "] "
                << "[" << toString(level) << "] ";
            (oss << ... << args);  // Logging the variable arguments
            std::string logMessage = oss.str();

            if (logFile_.is_open()) {
                logFile_ << logMessage << std::endl;
            } else {
                std::cout << logMessage << std::endl;
            }
        }
    }

private:
    Logger() : logLevel_(LogLevel::INFO) {}
    ~Logger() {
        if (logFile_.is_open()) {
            logFile_.close();
        }
    }

    // Get current time in a human-readable format
    std::string getCurrentTime() {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        std::ostringstream oss;
        oss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S")
            << '.' << std::setw(3) << std::setfill('0') << milliseconds.count();
        return oss.str();
    }

    // Convert log level enum to string
    std::string toString(LogLevel level) {
        switch (level) {
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::DEBUG: return "DEBUG";
            default: return "UNKNOWN";
        }
    }

    std::mutex mutex_;        // Mutex for thread-safe logging
    std::ofstream logFile_;   // File stream for logging to a file
    LogLevel logLevel_;       // Current log level for filtering messages

    // Deleting copy constructor and assignment operator
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
};

// Macros for convenient logging with worker details
#define LOG_INFO(workerIndex, workerType, ...) \
    Logger::getInstance().log(workerIndex, workerType, LogLevel::INFO, __VA_ARGS__)

#define LOG_WARNING(workerIndex, workerType, ...) \
    Logger::getInstance().log(workerIndex, workerType, LogLevel::WARNING, __VA_ARGS__)

#define LOG_ERROR(workerIndex, workerType, ...) \
    Logger::getInstance().log(workerIndex, workerType, LogLevel::ERROR, __VA_ARGS__)

#define LOG_DEBUG(workerIndex, workerType, ...) \
    Logger::getInstance().log(workerIndex, workerType, LogLevel::DEBUG, __VA_ARGS__)

#endif // LOGGER_HPP
