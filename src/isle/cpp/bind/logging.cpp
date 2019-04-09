#include "logging.hpp"

#include <functional>

#include <pybind11/pybind11.h>

// remove macro defined in termios.h on Mac to avoid clash in blaze
#ifdef VT1
  #undef VT1
#endif

#include "../cache.hpp"

namespace py = pybind11;

namespace isle {
    class Logger::Impl {
        // Get references the logger directly.
        py::object _pyLogger;

        // Get the individual functions only when needed.
        // Usually, each instance of Logger will only ever use
        // one or two of them.
#ifndef NDEBUG
        Cache<py::object, std::function<py::object()>> _pyDebug;
#endif
        Cache<py::object, std::function<py::object()>> _pyInfo;
        Cache<py::object, std::function<py::object()>> _pyWarning;
        Cache<py::object, std::function<py::object()>> _pyError;
        Cache<py::object, std::function<py::object()>> _pyCritical;

    public:
        explicit Impl(const std::string &name)
            : _pyLogger(py::module::import("logging").attr("getLogger")(name)),
#ifndef NDEBUG
              _pyDebug([impl=this](){ return impl->_pyLogger.attr("debug"); }),
#endif
              _pyInfo([impl=this](){ return impl->_pyLogger.attr("info"); }),
              _pyWarning([impl=this](){ return impl->_pyLogger.attr("warning"); }),
              _pyError([impl=this](){ return impl->_pyLogger.attr("error"); }),
              _pyCritical([impl=this](){ return impl->_pyLogger.attr("critical"); })
        { }

#ifndef NDEBUG
        inline void debug(const std::string &message) {
            _pyDebug.value()(message);
        }
#endif

        inline void info(const std::string &message) {
            _pyInfo.value()(message);
        }

        inline void warning(const std::string &message) {
            _pyWarning.value()(message);
        }

        inline void error(const std::string &message) {
            _pyError.value()(message);
        }

        inline void critical(const std::string &message) {
            _pyCritical.value()(message);
        }
    };


    Logger::Logger(const std::string &name)
        : pimpl(std::make_unique<Logger::Impl>(name))
    { }

    // Nothing special to do here.
    // But we need to keep the implementation hidden away
    // so we don't have to include pybind11 in the header.
    // std::unique_ptr needs to know the size of Logger::Impl.
    Logger::~Logger() = default;
    Logger::Logger(Logger &&) noexcept = default;
    Logger &Logger::operator=(Logger &&) noexcept = default;

#ifndef NDEBUG
    void Logger::debug(const std::string &message) {
        pimpl->debug(message);
    }
#endif

    void Logger::info(const std::string &message) {
        pimpl->info(message);
    }

    void Logger::warning(const std::string &message) {
        pimpl->warning(message);
    }

    void Logger::error(const std::string &message) {
        pimpl->error(message);
    }

    void Logger::critical(const std::string &message) {
        pimpl->critical(message);
    }
}  // namespace isle
