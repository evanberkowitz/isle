/** \file
 * \brief Bindings of Python's logging module to C++.
 *
 * This file is different from the other bindings as it exposes Python
 * objects to C++, not the other way around.
 * Use the class Logger for all terminal output in order to be consistent
 * with the Python part of Isle.
 * See the documentation in cli.py for more information.
 */

#ifndef BIND_LOGGING_HPP
#define BIND_LOGGING_HPP

#include <string>
#include <memory>

#include "../core.hpp"

namespace isle {
    /// Wrapper around Python's logging interface.
    /**
     * The member functions debug(), info(), warning(), error(), and critical()
     * all redirect to the functions with the same name in Python.
     * The only difference is that the C++ functions do not perform any automatic
     * formatting, the string has to be preformatted by the user.
     */
    class Logger {
        /// The actual implementation to keep pybing11 headers out of logging.hpp.
        class Impl;

        /// A PIMPL.
        std::unique_ptr<Impl> pimpl;

    public:
        /// Request a Python logger with the given name.
        explicit Logger(const std::string &name);

        Logger(const Logger &) = delete;
        Logger &operator=(const Logger &) = delete;

        // Implement those in .cpp because unique_ptr would need to know the
        // size of Impl here.
        Logger(Logger &&) noexcept;
        Logger &operator=(Logger &&) noexcept;
        // The d'tor of py::object is not noexcept.
        ~Logger();

#ifdef NDEBUG
        // Empty function that hopefully gets optimized away entirely.
        // Use a template to avoid any converions on the argument.
        template <typename T>
        constexpr void debug(const T &UNUSED(str)) noexcept { }

#else
        /// Log message with level DEBUG.
        /**
         * This function gets disabled if the macro `NDEBUG` is defined.
         * In that case, it should be optimized away entirely.
         */
        void debug(const std::string &message);
#endif

        /// Log message with level INFO.
        void info(const std::string &message);
        /// Log message with level WARNING.
        void warning(const std::string &message);
        /// Log message with level ERROR.
        void error(const std::string &message);
        /// Log message with level CRITICAL.
        void critical(const std::string &message);
    };

    /// Construct and return a new logger.
    inline Logger getLogger(const std::string &name) {
        return Logger(name);
    }
}  // namespace isle

#endif  // ndef BIND_LOGGING_HPP
