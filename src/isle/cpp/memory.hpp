#ifndef MEMORY_HPP
#define MEMORY_HPP

#include <cstddef>
#include <utility>

namespace isle {
//--------------------------------------------------------------------
//                      PointerFlagPair

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
     * \tparam T Type of pointee. Needs to satisfy the alignment restrictions
     *           described above.
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


//--------------------------------------------------------------------
//                      UnObHybridPtr

    /// Hybrid of unique and observer pointer.
    /**
     * Either owns some memory or observes it. This means that the object referenced
     * by this pointer is automatically deleted in the destructor of %UnObHybridPtr if
     * it is owned. If it is not owned no memory will be deallocated.
     *
     * The user selects which mode the hybrid pointer is in and is responsible for
     * choosing the right one. The object referenced by the hybrid pointer must be
     * managed by the user if the hybrid pointer does not own it.
     *
     * Using an %UnObHybridPtr is only slightly safer than using a raw pointer.
     * So special care has to be taken when choosing it and standard smart pointers
     * should be preferred when possible.
     *
     * %UnObHybridPtr can not be copied in order to allow for unique references.
     *
     * \tparam Et Type of the pointee. Needs to satisfy the alignment restrictions
     *            of the pointee of a PointerFlagPair, i.e. must be aligned to a
     *            boundary larger than 1.
     *
     * \see `makeUnObHybrid` for a helper function to construct owning hybrid pointers.
     */
    template <typename ET>
    struct UnObHybridPtr {
        using ElementType = ET;  ///< Type of pointee.
        using Pointer = ET*;     ///< Type of raw pointer.

        /// Set to `nullptr`.
        UnObHybridPtr() noexcept : _pfp{nullptr} { }

        /// Construct from raw pointer and set whether memory is owned.
        UnObHybridPtr(const Pointer ptr, const bool owned) noexcept
            : _pfp{ptr, owned} { }

        /// Copying is not allowed.
        UnObHybridPtr(const UnObHybridPtr&) = delete;
        /// Copying is not allowed.
        UnObHybridPtr &operator=(const UnObHybridPtr&) = delete;

        /// Move construct and set other to `nullptr`.
        UnObHybridPtr(UnObHybridPtr &&other) noexcept
            : _pfp{std::exchange(other._pfp, nullptr)} { }

        /// Move assign and set other to `nullptr`.
        UnObHybridPtr &operator=(UnObHybridPtr &&other) noexcept {
            reset(other._pfp);
            other._pfp = nullptr;
            return *this;
        }

        ~UnObHybridPtr() noexcept {
            if (owns())
                _pfp.free();
        }

        /// Swap with other pointer.
        void swap(UnObHybridPtr &other) noexcept {
            swap(_pfp, other._pfp);
        }

        /// Set from a new pointer and free old memory if owned.
        void reset(const Pointer p, const bool owned) noexcept {
            if (owns())
                _pfp.free();
            _pfp = _PtrFlag{p, owned};
        }

        /// Set to `nullptr` and free old memory if owned.
        void reset(const std::nullptr_t p = nullptr) noexcept {
            if (owns())
                _pfp.free();
            _pfp = p;
        }

        /// Get raw pointer to managed object, `nullptr` if nothing owned.
        Pointer get() const noexcept {
            return _pfp.pointer();
        }

        /// Retrun whether current memory is owned; if nothing owned, returns `false`.
        bool owns() const noexcept {
            return _pfp.flag();
        }

        /// Return raw pointer to managed object and set %UnObHybridPtr to `nullptr`.
        Pointer release() noexcept {
            return std::exchange(_pfp, nullptr).pointer();
        }

        /// `true` if an object is managed, `false` otherwise.
        explicit operator bool() const noexcept {
            return _pfp.pointer();
        }

        /// Dereference pointer, i.e. return managed object.
        std::add_lvalue_reference_t<ElementType> &operator*() const {

            return *_pfp.pointer();
        }

        /// Return pointer to managed object.
        Pointer operator->() const noexcept {
            return _pfp.pointer();
        }

    private:
        using _PtrFlag = PointerFlagPair<ET>;  ///< Type to hold pointer and flag.

        _PtrFlag _pfp;  ///< Combined pointer to managed object and ownership flag.

        /// Reset using a _PtrFlag object.
        void _reset(const _PtrFlag pfp) {
            if (owns())
                _pfp.free();
            _pfp = pfp;
        }
    };

    /// Swap two UnObHybridPtrs.
    template <typename ET>
    void swap(UnObHybridPtr<ET> &lhs, UnObHybridPtr<ET> &rhs) noexcept {
        lhs.swap(rhs);
    }

    /// Make an UnObHybridPtr of given pointee type.
    /**
     * Allocated memory for and constructs a new object of type `ET` with given
     * parameters and an UnObHybridPtr which owns the allocated memory.
     *
     * \tparam ET Element type of the new UnObHybridPtr, i.e. type of pointee.
     * \param args Parameters to forward to constructor of managed object of type `ET`
     * \returns New UnObHybridPtr owning a new instance of type `ET`.
     */
    template <typename ET, typename... Args>
    UnObHybridPtr<ET> makeUnObHybrid(Args&&... args) {
        return UnObHybridPtr<ET>{new ET(std::forward<Args>(args)...), true};
    }

}  // namespace isle

#endif  // ndef MEMORY_HPP
