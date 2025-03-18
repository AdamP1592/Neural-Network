#ifndef LOGGER_H
#define LOGGER_H

#include <fstream>
#include <string>
#include <mutex>
#include <stdexcept>

class Logger {
public:
    // Log a message to the default log file.
    static void log(const std::string &message) {
        std::lock_guard<std::mutex> lock(mtx);
        // Open the file if it isn't already open.
        if (!logFile.is_open()) {
            logFile.open(filePath, std::ios::out | std::ios::app);
            if (!logFile) {
                throw std::runtime_error("Failed to open log file: " + filePath);
            }
        }
        logFile << message << std::endl;
    }
    
    // Optionally, change the default file path at runtime.
    static void setFilePath(const std::string &newFilePath) {
        std::lock_guard<std::mutex> lock(mtx);
        if (logFile.is_open()) {
            logFile.close();
        }
        filePath = newFilePath;
    }

private:
    // Default log file path.
    inline static std::string filePath = "./log/default_log.txt";
    inline static std::ofstream logFile;
    inline static std::mutex mtx;
};

#endif // LOGGER_H
