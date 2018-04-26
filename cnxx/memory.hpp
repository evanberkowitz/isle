#ifndef MEMORY_HPP
#define MEMORY_HPP

#include <cstddef>
#include <utility>

namespace cnxx {
    /// Hold a pointer and a flag in one piece of memory.
    /**
     * Uses the least significant bit of a pointer to store a boolean flag.
     * This requires that the pointee type `T` has an alignment
     * greater than one byte such that the least significant bit of a pointer
     * `T*` is always zero. Otherwise, the last bit of the pointer is needed to address
     * the pointee. This constraint is enforced at compile time.
     *
     * \warning
     * Special care has to be taken that no objects are referenced which are not located
     * at their usual alignment boundary. Otherwise the flag might corrupt the pointer
     * irreversibly.
     *
     * \note `%PointerFlagPair` does not own any ressources. The user is responsible for
     *       de-/allocating memory.
     *
     * \tparam T Type of pointee.
     *
     * This obviously is complete overkill here but hey, why not? :D
     */
    template <typename T>
    struct PointerFlagPair {
        static_assert(alignof(T) > 1,
                      "PointerFlagPair not supported for types with 1-byte alignment,"
                      " need to have zero bits in pointer");

        using ElementType = T;  ///< Type of pointee.
        using Pointer = T*;     ///< Type of raw pointer.
        using Flag = bool;      ///< Type of flag.

        /// Construct from raw pointer and flag.
        explicit PointerFlagPair(const Pointer ptr, const Flag flag = false) noexcept
            : _ptr{reinterpret_cast<_IntPtr>(ptr) | flag} { }

        /// Construct form nullptr, sets flag to false.
        explicit PointerFlagPair(const std::nullptr_t = nullptr) noexcept
            : _ptr{0} { }

        /// Assign from nullptr, sets flag to false.
        PointerFlagPair &operator=(const std::nullptr_t) noexcept {
            _ptr = 0;
            return *this;
        }

        /// Copy construct.
        PointerFlagPair(const PointerFlagPair &) noexcept = default;
        /// Copy assign.
        PointerFlagPair &operator=(const PointerFlagPair &) noexcept = default;

        /// Move construct.
        PointerFlagPair(PointerFlagPair &&other) noexcept
            : _ptr{std::exchange(other._ptr, 0)} { }
        /// Move assign.
        PointerFlagPair &operator=(PointerFlagPair &&other) noexcept {
            _ptr = std::exchange(other._ptr, 0);
            return *this;
        }

        ~PointerFlagPair() = default;

        /// Swap with other %PointerFlagPair.
        void swap(PointerFlagPair &other) noexcept {
            std::swap(_ptr, other._ptr);
        }

        /// Set flag.
        void flag(const Flag flag) noexcept {
            _ptr |= flag;
        }

        /// Get flag.
        Flag flag() const noexcept {
            return _ptr & _flagMask;
        }

        /// Set pointer; does not free memory.
        void pointer(const Pointer ptr) noexcept {
            _ptr = reinterpret_cast<_IntPtr>(ptr) | flag();
        }

        /// Get pointer,
        Pointer pointer() const noexcept {
            return reinterpret_cast<Pointer>(_ptr & _ptrMask);
        }

        /// Delete pointer.
        void free() noexcept {
            delete pointer();
        }

    private:
        /// Integer stand-in for raw pointer plus flag to allow for bitwise ops
        using _IntPtr = std::size_t;

        _IntPtr _ptr;  ///< Combined pointer and flag.

        static constexpr std::size_t _flagMask = 1u; ///< Bit mask to extract the flag.
        static constexpr std::size_t _ptrMask = ~_flagMask; ///< Bit mask to extract the pointer.
    };

    /// Swap two PointerFlagPairs.
    template <typename T>
    void swap(PointerFlagPair<T> &lhs, PointerFlagPair<T> &rhs) noexcept {
        lhs.swap(rhs);
    }

}  // namespace cnxx

#endif  // ndef MEMORY_HPP
