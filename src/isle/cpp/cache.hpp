/** \file
 * \brief Cache objects on demand.
 */

#ifndef CACHE_HPP
#define CACHE_HPP

#include <type_traits>
#include <utility>

namespace isle {
    /// Construct an object on demand and memoize it.
    /**
     * This is a container for a single object ('item') and an associated
     * generator to construct this object. When access to the item is requested
     * and the cache is invalid ('empty'), the generator is invoked to construct
     * an item. Hence Cache provides a way to store objects that are expensive to
     * make but will only construct them if and when they are first needed.
     *
     * Note that both the stored object and generator are part of the memory
     * footprint of Cache.
     *
     * \tparam IT Type of the object ('item') to store.
     *            <B>Must be default constructible.</B>
     * \tparam GT Type of the generator to construct items.
     *            Invoking the generator (`operator()`) has to return
     *            a valid item of type IT.
     *
     * /see makeCache() for a way to deduce the template parameter.
     */
    template <typename IT, typename GT>
    class Cache {
    public:
        /// Construct an item in place.
        /**
         * /param gen Generator to use, is copied in.
         * /param itemArgs Arguments to forward to the constructor of the item.
         */
        template <typename... ItemArgs>
        Cache(const GT &gen, ItemArgs&& ...itemArgs)
            : _item{std::forward<ItemArgs>(itemArgs)...}, _valid{true},
              _gen(gen) { }

        /// Construct an item in place.
        /**
         * /param gen Generator to use.
         * /param itemArgs Arguments to forward to the constructor of the item.
         */
        template <typename... ItemArgs>
        Cache(const GT &&gen, ItemArgs&& ...itemArgs)
            : _item{std::forward<ItemArgs>(itemArgs)...}, _valid{true},
              _gen(std::move(gen)) { }

        /// Default initialize item.
        /**
         * \param gen Generator to use, is copied in.
         */
        explicit Cache(const GT &gen)
            : _item{}, _valid{false}, _gen(gen) { }

        /// Default initialize item.
        /**
         * \param gen Generator to use.
         */
        explicit Cache(GT &&gen)
            : _item{}, _valid{false}, _gen(std::move(gen)) { }

        Cache(const Cache &other) = default;
        Cache &operator=(const Cache &other) = default;
        Cache(Cache &&other)
            noexcept(std::is_nothrow_move_constructible<IT>::value
                     && std::is_nothrow_move_constructible<GT>::value) = default;
        Cache &operator=(Cache &&other)
            noexcept(std::is_nothrow_move_assignable<IT>::value
                     && std::is_nothrow_move_assignable<GT>::value) = default;
        ~Cache() = default;

        /// Invalidate the cache, item is reconstructed on next use.
        void invalidate() noexcept {
            _valid = false;
        }

        /// Reset the item to a default constructed state and invalidate.
        void clear() {
            _item = IT{};
            invalidate();
        }

        /// Access the item, construct if cache is invalid.
        operator IT&() {
            _constructIfNeeded();
            return _item;
        }

        /// Access the item, construct if cache is invalid.
        operator const IT&() const {
            _constructIfNeeded();
            return _item;
        }

        /// Access the item, construct if cache is invalid.
        IT &value() {
            _constructIfNeeded();
            return _item;
        }

        /// Access the item, construct if cache is invalid.
        const IT &value() const {
            _constructIfNeeded();
            return _item;
        }


    private:
        mutable IT _item;  ///< The stored object.
        mutable bool _valid;  ///< If `false`, _item needs to be constructed before use.
        GT _gen;  ///< Generator to construct _item when needed.

        /// Construct _item if `_valid==false`
        void _constructIfNeeded() const {
            if (!_valid) {
                _item = _gen();
                _valid = true;
            }
        }
    };

    /// Make a new cache with given generator and constructs the item with given arguments.
    template <typename GT, typename... ItemArgs>
    auto makeCache(GT &&gen, ItemArgs&& ...itemArgs) {
        return Cache<std::decay_t<decltype(std::declval<GT>()())>,
                     std::decay_t<GT>>{
                         std::forward<GT>(gen),
                         std::forward<ItemArgs>(itemArgs)...
                         };
    }
}  // namespace isle

#endif  // ndef CACHE_HPP
