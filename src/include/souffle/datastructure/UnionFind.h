/*
 * Souffle - A Datalog Compiler
 * Copyright (c) 2017 The Souffle Developers. All rights reserved
 * Licensed under the Universal Permissive License v 1.0 as shown at:
 * - https://opensource.org/licenses/UPL
 * - <souffle root>/licenses/SOUFFLE-UPL.txt
 */

/************************************************************************
 *
 * @file UnionFind.h
 *
 * Defines a union-find data-structure
 *
 ***********************************************************************/

#pragma once

#include "souffle/datastructure/LambdaBTree.h"
#include "souffle/datastructure/PiggyList.h"
#include "souffle/utility/MiscUtil.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

namespace souffle {

// branch predictor hacks
#define unlikely(x) __builtin_expect((x), 0)
#define likely(x) __builtin_expect((x), 1)

// number of bits that the rank is
constexpr uint8_t split_size = 8u;

struct rank_t {
public:
    rank_t(uint8_t val) : value_(val) {}

    explicit operator uint64_t() const noexcept {
        return value_;
    }

    int operator==(rank_t x) const {
        return x.value_ == value_;
    }

    int operator!=(rank_t x) const {
        return x.value_ != value_;
    }

    int operator<(rank_t x) const {
        return x.value_ < value_;
    }

    int operator>(rank_t x) const {
        return x.value_ > value_;
    }

    rank_t succ() {
        return rank_t(value_ + 1);
    }

private:
    uint8_t value_;
};

/* technically uint56_t, but, doesn't exist. Just be careful about storing > 2^56 elements. */
struct parent_t {
public:
    explicit parent_t(uint64_t val) : value_(val) {}

    int operator==(parent_t x) const {
        return x.value_ == value_;
    }

    int operator!=(parent_t x) const {
        return x.value_ != value_;
    }

    int operator<(parent_t x) const {
        return x.value_ < value_;
    }

    int operator>(parent_t x) const {
        return x.value_ > value_;
    }

    explicit operator uint64_t() const noexcept {
        return value_;
    }

private:
    uint64_t value_;
};

// block_t stores parent in the upper half, rank in the lower half
struct block_t {
public:
    // explicit operator int() const noexcept {
    //   return value_;
    // }
    int operator==(block_t x) const {
        return x.value_ == value_;
    }

    int operator!=(block_t x) const {
        return x.value_ != value_;
    }

    static block_t pr2b(const parent_t parent, const rank_t rank) {
        block_t block{((((uint64_t)parent) << split_size) | (uint64_t)rank)};
        return block;
    };

    /**
     * Extract parent from block
     * @param inblock the block to be masked
     * @return The parent_t contained in the upper half of block_t
     */
    parent_t b2p() const {
        return parent_t(value_ >> split_size);
    };

    /**
     * Extract rank from block
     * @param inblock the block to be masked
     * @return the rank_t contained in the lower half of block_t
     */
    rank_t b2r() const {
        return rank_t(value_ & rank_mask);
    };

    // This must be exposed for use of std::atomic<block_t>
    uint64_t value_;

private:
    // block_t & rank_mask extracts the rank
    static constexpr uint64_t rank_mask = (1ul << split_size) - 1;
};

static_assert(std::is_trivially_copyable<block_t>::value);

static_assert(std::is_copy_constructible<block_t>::value);

static_assert(std::is_move_constructible<block_t>::value);

static_assert(std::is_copy_assignable<block_t>::value);

static_assert(std::is_move_assignable<block_t>::value);

/**
 * Structure that emulates a Disjoint Set, i.e. a data structure that supports efficient union-find operations
 */
class DisjointSet {
    template <typename TupleType>
    friend class EquivalenceRelation;

    PiggyList<std::atomic<block_t>> a_blocks;

public:
    DisjointSet() = default;

    // copy ctor
    DisjointSet(DisjointSet& other) = delete;
    // move ctor
    DisjointSet(DisjointSet&& other) = delete;

    // copy assign ctor
    DisjointSet& operator=(DisjointSet& ds) = delete;
    // move assign ctor
    DisjointSet& operator=(DisjointSet&& ds) = delete;

    /**
     * Return the number of elements in this disjoint set (not the number of pairs)
     */
    inline std::size_t size() {
        auto sz = a_blocks.size();
        return sz;
    };

    /**
     * Yield reference to the node by its node index
     * @param node node to be searched
     * @return the parent block of the specified node
     */
    inline std::atomic<block_t>& get(parent_t node) const {
        auto& ret = a_blocks.get((std::size_t)node);
        return ret;
    };

    /**
     * Equivalent to the find() function in union/find
     * Find the highest ancestor of the provided node - flattening as we go
     * @param x the node to find the parent of, whilst flattening its set-tree
     * @return The parent of x
     */
    parent_t findNode(parent_t x) {
        // while x's parent is not itself
        while (x != b2p(get(x))) {
            block_t xState = get(x);
            // yield x's parent's parent
            parent_t newParent = b2p(get(b2p(xState)));
            // construct block out of the original rank and the new parent
            block_t newState = pr2b(newParent, b2r(xState));

            this->get(x).compare_exchange_strong(xState, newState);

            x = newParent;
        }
        return x;
    }

private:
    /**
     * Update the root of the tree of which x is, to have y as the base instead
     * @param x : old root
     * @param oldrank : old root rank
     * @param y : new root
     * @param newrank : new root rank
     * @return Whether the update succeeded (fails if another root update/union has been perfomed in the
     * interim)
     */
    bool updateRoot(const parent_t x, const rank_t oldrank, const parent_t y, const rank_t newrank) {
        block_t oldState = get(x);
        parent_t nextN = b2p(oldState);
        rank_t rankN = b2r(oldState);

        if (nextN != x || rankN != oldrank) return false;
        // set the parent and rank of the new record
        block_t newVal = pr2b(y, newrank);

        return this->get(x).compare_exchange_strong(oldState, newVal);
    }

public:
    /**
     * Clears the DisjointSet of all nodes
     * Invalidates all iterators
     */
    void clear() {
        a_blocks.clear();
    }

    /**
     * Check whether the two indices are in the same set
     * @param x node to be checked
     * @param y node to be checked
     * @return where the two indices are in the same set
     */
    bool sameSet(parent_t x, parent_t y) {
        while (true) {
            x = findNode(x);
            y = findNode(y);
            if (x == y) return true;
            // if x's parent is itself, they are not the same set
            if (b2p(get(x)) == x) return false;
        }
    }

    /**
     * Union the two specified index nodes
     * @param x node to be unioned
     * @param y node to be unioned
     */
    void unionNodes(parent_t x, parent_t y) {
        while (true) {
            x = findNode(x);
            y = findNode(y);

            // no need to union if both already in same set
            if (x == y) return;

            rank_t xrank = b2r(get(x));
            rank_t yrank = b2r(get(y));

            // if x comes before y (better rank or earlier & equal node)
            if (xrank > yrank || ((xrank == yrank) && x > y)) {
                std::swap(x, y);
                std::swap(xrank, yrank);
            }
            // join the trees together
            // perhaps we can optimise the use of compare_exchange_strong here, as we're in a pessimistic loop
            if (!updateRoot(x, xrank, y, yrank)) {
                continue;
            }
            // make sure that the ranks are orderable
            if (xrank == yrank) {
                updateRoot(y, yrank, y, yrank.succ());
            }
            break;
        }
    }

    /**
     * Create a node with its parent as itself, rank 0
     * @return the newly created block
     */
    inline block_t makeNode() {
        // make node and find out where we've added it
        std::size_t nodeDetails = a_blocks.createNode();

        a_blocks.get(nodeDetails).store(pr2b(parent_t(nodeDetails), 0));

        return a_blocks.get(nodeDetails).load();
    };

    /**
     * Extract parent from block
     * @param inblock the block to be masked
     * @return The parent_t contained in the upper half of block_t
     */
    static inline parent_t b2p(const block_t inblock) {
        return inblock.b2p();
    };

    /**
     * Extract rank from block
     * @param inblock the block to be masked
     * @return the rank_t contained in the lower half of block_t
     */
    static inline rank_t b2r(const block_t inblock) {
        return inblock.b2r();
    };

    /**
     * Yield a block given parent and rank
     * @param parent the top half bits
     * @param rank the lower half bits
     * @return the resultant block after merge
     */
    static inline block_t pr2b(const parent_t parent, const rank_t rank) {
        return block_t::pr2b(parent, rank);
    };
};

template <typename StorePair>
struct EqrelMapComparator {
    int operator()(const StorePair& a, const StorePair& b) {
        if (a.first < b.first) {
            return -1;
        } else if (b.first < a.first) {
            return 1;
        } else {
            return 0;
        }
    }

    bool less(const StorePair& a, const StorePair& b) {
        return operator()(a, b) < 0;
    }

    bool equal(const StorePair& a, const StorePair& b) {
        return operator()(a, b) == 0;
    }
};

template <typename SparseDomain>
class SparseDisjointSet {
    DisjointSet ds;

    template <typename TupleType>
    friend class EquivalenceRelation;

    using PairStore = std::pair<SparseDomain, parent_t>;
    using SparseMap =
            LambdaBTreeSet<PairStore, std::function<parent_t(PairStore&)>, EqrelMapComparator<PairStore>>;
    using DenseMap = RandomInsertPiggyList<SparseDomain>;

    typename SparseMap::operation_hints last_ins;

    SparseMap sparseToDenseMap;
    // mapping from union-find val to souffle, union-find encoded as index
    DenseMap denseToSparseMap;

public:
    /**
     * Retrieve dense encoding, adding it in if non-existent
     * @param in the sparse value
     * @return the corresponding dense value
     */
    parent_t toDense(const SparseDomain in) {
        // insert into the mapping - if the key doesn't exist (in), the function will be called
        // and a dense value will be created for it
        PairStore p = {in, -1};
        return sparseToDenseMap.insert(p, [&](PairStore& p) {
            parent_t c2 = DisjointSet::b2p(this->ds.makeNode());
            this->denseToSparseMap.insertAt(c2, p.first);
            p.second = c2;
            return c2;
        });
    }

public:
    SparseDisjointSet() = default;

    // copy ctor
    SparseDisjointSet(SparseDisjointSet& other) = delete;

    // move ctor
    SparseDisjointSet(SparseDisjointSet&& other) = delete;

    // copy assign ctor
    SparseDisjointSet& operator=(SparseDisjointSet& other) = delete;

    // move assign ctor
    SparseDisjointSet& operator=(SparseDisjointSet&& other) = delete;

    /**
     * For the given dense value, return the associated sparse value
     *   Undefined behaviour if dense value not in set
     * @param in the supplied dense value
     * @return the sparse value from the denseToSparseMap
     */
    inline const SparseDomain toSparse(const parent_t in) const {
        return denseToSparseMap.get(in);
    };

    /* a wrapper to enable checking in the sparse set - however also adds them if not already existing */
    inline bool sameSet(SparseDomain x, SparseDomain y) {
        return ds.sameSet(toDense(x), toDense(y));
    };
    /* finds the node in the underlying disjoint set, adding the node if non-existent */
    inline SparseDomain findNode(SparseDomain x) {
        return toSparse(ds.findNode(toDense(x)));
    };
    /* union the nodes, add if not existing */
    inline void unionNodes(SparseDomain x, SparseDomain y) {
        ds.unionNodes(toDense(x), toDense(y));
    };

    inline std::size_t size() {
        return ds.size();
    };

    /**
     * Remove all elements from this disjoint set
     */
    void clear() {
        ds.clear();
        sparseToDenseMap.clear();
        denseToSparseMap.clear();
    }

    /* wrapper for node creation */
    inline void makeNode(SparseDomain val) {
        // dense has the behaviour of creating if not exists.
        toDense(val);
    };

    /* whether the supplied node exists */
    inline bool nodeExists(const SparseDomain val) const {
        return sparseToDenseMap.contains({val, -1});
    };

    inline bool contains(SparseDomain v1, SparseDomain v2) {
        if (nodeExists(v1) && nodeExists(v2)) {
            return sameSet(v1, v2);
        }
        return false;
    }
};
}  // namespace souffle
