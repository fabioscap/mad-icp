#pragma once

#include "utils.h"

#include <Eigen/Core>
#include <atomic>
#include <iostream>
#include <memory>
#include <vector>

namespace flib {
  using IdxType = size_t;

  // wrap nodes storing also the address of the storage. We don't want this
  // information in the nodes because if we serialize the tree then all of the
  // pointers become invalid

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

    struct NodeHandle {
      IdxType idx;
      std::vector<typename BTreeVector<T>::Node>* storage_ptr;

      NodeHandle(IdxType i, std::vector<typename BTreeVector<T>::Node>* ptr) : idx{i}, storage_ptr{ptr} {
      }

      bool has_left() const {
        return storage_ptr->operator[](idx).left != BTreeVector<T>::no_idx;
      }

      bool has_right() const {
        return storage_ptr->operator[](idx).right != BTreeVector<T>::no_idx;
      }
      NodeHandle left() const {
        return NodeHandle{storage_ptr->operator[](idx).left, storage_ptr};
      }

      NodeHandle right() const {
        return NodeHandle{storage_ptr->operator[](idx).right, storage_ptr};
      }

      template <typename... Args>
      IdxType add_left(Args&&... args) {
        if (has_left()) {
          left().node().data = T(std::forward<Args>(args)...);
        } else {
          storage_ptr->emplace_back(idx, std::forward<Args>(args)...);
          node().left = storage_ptr->size() - 1;
        }

        return node().left;
      }

      template <typename... Args>
      IdxType add_right(Args&&... args) {
        if (has_right()) {
          right().node().data = T(std::forward<Args>(args)...);
        } else {
          storage_ptr->emplace_back(idx, std::forward<Args>(args)...);
          node().right = storage_ptr->size() - 1;
        }

        return node().right;
      }

      typename BTreeVector<T>::Node& node() {
        return storage_ptr->operator[](idx);
      }
      typename BTreeVector<T>::Node& node() const {
        return storage_ptr->operator[](idx);
      }

      void print_tree(const std::string& prefix = "", bool is_left = true) const {
        if (this->has_right()) {
          right().print_tree(prefix + (is_left ? "│   " : "    "), false);
        }

        std::cout << prefix << (is_left ? "└── " : "┌── ") << storage_ptr->operator[](idx).data << "\n";

        if (this->has_left()) {
          left().print_tree(prefix + (is_left ? "    " : "│   "), true);
        }
      }

    protected:
      // be careful, no access control
      typename BTreeVector<T>::Node& node(IdxType idx) {
        return storage_ptr->operator[](idx);
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

  struct MADNodeHandle : BaseType::NodeHandle {
    using BaseType::NodeHandle::NodeHandle;
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
};

// TODOs
// 1- Async construction
// 2- The node handle should have a pointer to the Tree and not the storage. So I can precompute
// the leaves and save them somewhere (perhaps madtree has a std::vector of leaves)
// 3- fix the weird interface of nodehandles perhaps with CRTP so that left() returns the correct
// inherited type. this is ugly: MADNodeHandle{nh.node().left, storage_ptr}

inline std::ostream& operator<<(std::ostream& os, const MADnode& p) {
  os << p.mean.transpose();
  return os;
}
