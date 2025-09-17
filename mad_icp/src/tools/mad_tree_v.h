#pragma once

#include "utils.h"

#include <Eigen/Core>
#include <atomic>
#include <iostream>
#include <list>
#include <memory>

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
      void add_left(Args&&... args) {
        if (has_left()) {
          left().node().data = T(std::forward<Args>(args)...);
        } else {
          storage_ptr->emplace_back(idx, std::forward<Args>(args)...);
          node().left = storage_ptr->size() - 1;
        }
      }

      template <typename... Args>
      void add_right(Args&&... args) {
        if (has_right()) {
          right().node().data = T(std::forward<Args>(args)...);
        } else {
          storage_ptr->emplace_back(idx, std::forward<Args>(args)...);
          node().right = storage_ptr->size() - 1;
        }
      }

      typename BTreeVector<T>::Node& node() {
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
    };

    NodeHandle root() {
      return NodeHandle{
        0,
        &storage,
      };
    }
  };

} // namespace flib

struct MADnode {
  using IdxType = flib::IdxType;
  int num_points_;
  bool matched_;
  Eigen::Vector3d mean_;
  Eigen::Vector3d bbox_;
  Eigen::Matrix3d eigenvectors_;
};

using ContainerType    = std::vector<Eigen::Vector3d>;
using ContainerTypePtr = ContainerType*;
using IteratorType     = typename ContainerType::iterator;
using LeafList         = std::vector<MADnode*>;

struct MADtreeV : flib::BTreeVector<MADnode> {
  using IdxType  = flib::IdxType;
  using BaseType = flib::BTreeVector<MADnode>;

  struct MADNodeHandle : BaseType::NodeHandle {
    MADNodeHandle(const ContainerTypePtr vec,
                  const IteratorType begin,
                  const IteratorType end,
                  const double b_max,
                  const double b_min,
                  const int level,
                  const int max_parallel_level,
                  IdxType* parent,
                  IdxType* plane_predecessor);
  };

  MADNodeHandle root() {
    return MADNodeHandle{
      0,
      &storage,
    };
  }
};