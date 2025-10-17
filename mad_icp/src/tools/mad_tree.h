#pragma once

#include "utils.h"

#include <Eigen/Core>
#include <atomic>
#include <iostream>
#include <memory>
#include <vector>

namespace flib {
  using IdxType = size_t;

  template <typename T, size_t NS = 1024>
  struct BTreeVector {
    static constexpr IdxType no_idx = static_cast<IdxType>(-1);
    struct Node {
      IdxType left{no_idx};
      IdxType right{no_idx};
      IdxType parent{no_idx};

      T data;
      template <typename... Args>
      Node(Args&&... args) : data(std::forward<Args>(args)...) {
      }

      template <typename... Args>
      Node(IdxType p, Args&&... args) : parent(p), data(std::forward<Args>(args)...) {
      }
    };

    std::vector<Node> storage;

    BTreeVector() {
      storage.reserve(NS);
    }

    template <typename... Args>
    BTreeVector(Args&&... args) {
      storage.reserve(NS);
      storage.emplace_back(std::forward<Args>(args)...);
    }

    // wrap nodes storing also the address of the storage. We don't want this
    // information in the nodes because if we serialize the tree then all of the
    // pointers become invalid
    template <typename Derived>
    struct NodeHandle {
      IdxType idx;
      std::vector<typename BTreeVector<T>::Node>& storage;

      NodeHandle(IdxType i, std::vector<typename BTreeVector<T>::Node>& str) : idx{i}, storage{str} {
      }

      bool has_left() const {
        return storage[idx].left != BTreeVector<T>::no_idx;
      }

      bool has_right() const {
        return storage[idx].right != BTreeVector<T>::no_idx;
      }
      Derived left() const {
        return Derived{storage[idx].left, storage};
      }

      Derived right() const {
        return Derived{storage[idx].right, storage};
      }

      template <typename... Args>
      Derived add_left(Args&&... args) {
        if (has_left()) {
          left().node().data = T(std::forward<Args>(args)...);
        } else {
          storage.emplace_back(idx, std::forward<Args>(args)...);
          node().left = storage.size() - 1;
        }

        return Derived{node().left, storage};
      }

      template <typename... Args>
      Derived add_right(Args&&... args) {
        if (has_right()) {
          right().node().data = T(std::forward<Args>(args)...);
        } else {
          storage.emplace_back(idx, std::forward<Args>(args)...);
          node().right = storage.size() - 1;
        }

        return Derived{node().right, storage};
      }

      typename BTreeVector<T>::Node& node() {
        return storage[idx];
      }
      typename BTreeVector<T>::Node& node() const {
        return storage[idx];
      }

      void print_tree(const std::string& prefix = "", bool is_left = true) const {
        if (this->has_right()) {
          right().print_tree(prefix + (is_left ? "│   " : "    "), false);
        }

        std::cout << prefix << (is_left ? "└── " : "┌── ") << storage[idx].data << "\n";

        if (this->has_left()) {
          left().print_tree(prefix + (is_left ? "    " : "│   "), true);
        }
      }

    protected:
      // be careful, no access control
      typename BTreeVector<T>::Node& node(IdxType idx) {
        return storage[idx];
      }
    };
  };

} // namespace flib

struct MADnode {
  using IdxType = flib::IdxType;
  int num_points;
  bool matched;
  Eigen::Vector3d mean;
  Eigen::Vector3d bbox;
  Eigen::Matrix3d eigenvectors;
};

using ContainerType    = std::vector<Eigen::Vector3d>;
using ContainerTypePtr = ContainerType*;
using IteratorType     = typename ContainerType::iterator;
using LeafList         = std::vector<MADnode*>;

struct MADtreeV : flib::BTreeVector<MADnode> {
  using IdxType  = flib::IdxType;
  using BaseType = flib::BTreeVector<MADnode>;

  struct MADNodeHandle : BaseType::NodeHandle<MADNodeHandle> {
    using BaseType::NodeHandle<MADNodeHandle>::NodeHandle;
    void build(const ContainerTypePtr vec,
               const IteratorType begin,
               const IteratorType end,
               const double b_max,
               const double b_min,
               const int level,
               const int max_parallel_level,
               IdxType parent            = no_idx,
               IdxType plane_predecessor = no_idx);

    void apply_transform(const Eigen::Matrix3d& r, const Eigen::Vector3d& t);

    const MADnode best_matching_leaf_fast(const Eigen::Vector3d& query) const;

    void get_leaves(std::back_insert_iterator<LeafList> it);
  };

  // returns handle to root
  MADNodeHandle build(const ContainerTypePtr vec,
                      const IteratorType begin,
                      const IteratorType end,
                      const double b_max,
                      const double b_min,
                      const int max_parallel_level);

  auto root() {
    return MADNodeHandle{0, storage};
  }
};

// TODOs
// 1- Async construction (probably hard due to many threads that can write in the same memory)
// 2- The node handle should have a pointer to the Tree and not the storage. So I can precompute
// the leaves and save them somewhere (perhaps madtree has a std::vector of leaves)

inline std::ostream& operator<<(std::ostream& os, const MADnode& p) {
  os << p.mean.transpose();
  return os;
}
