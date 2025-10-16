#include "mad_tree.h"

#include <Eigen/Eigenvalues>

MADtreeV::MADNodeHandle MADtreeV::build(const ContainerTypePtr vec,
                                        const IteratorType begin,
                                        const IteratorType end,
                                        const double b_max,
                                        const double b_min,
                                        const int max_parallel_level) {
  // build the root
  storage.emplace_back();

  MADNodeHandle root{0, &storage};

  root.build(vec, begin, end, b_max, b_min, 0, max_parallel_level, no_idx, no_idx);

  return root;
}

void MADtreeV::MADNodeHandle::build(const ContainerTypePtr vec,
                                    const IteratorType begin,
                                    const IteratorType end,
                                    const double b_max,
                                    const double b_min,
                                    const int level,
                                    const int max_parallel_level,
                                    IdxType parent,
                                    IdxType plane_predecessor) {
  auto& n = node();
  auto& d = n.data;

  n.parent = parent;

  Eigen::Matrix3d cov;
  computeMeanAndCovariance(d.mean, cov, begin, end);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es;
  es.computeDirect(cov);

  d.eigenvectors = es.eigenvectors();

  d.num_points = computeBoundingBox(d.bbox, d.mean, d.eigenvectors.transpose(), begin, end);

  auto nr = std::distance(begin, end);

  if (d.bbox(2) < b_max) {
    if (plane_predecessor != no_idx) {
      d.eigenvectors.setZero();
      d.eigenvectors.col(0) = node(plane_predecessor).data.eigenvectors.col(0);
    } else {
      if (d.num_points < 3) {
        Node* node_ptr = &n;
        while (node_ptr->parent != no_idx && node_ptr->data.num_points < 3)
          node_ptr = &node(node_ptr->parent);
        d.eigenvectors.setZero();
        d.eigenvectors.col(0) = node_ptr->data.eigenvectors.col(0);
      }
    }

    Eigen::Vector3d& nearest_point = *begin;
    double shortest_dist           = std::numeric_limits<double>::max();
    for (IteratorType it = begin; it != end; ++it) {
      const Eigen::Vector3d& v = *it;
      const double dist        = (v - d.mean).norm();
      if (dist < shortest_dist) {
        nearest_point = v;
        shortest_dist = dist;
      }
    }
    d.mean = nearest_point;

    return;
  }
  if (plane_predecessor == no_idx) {
    if (d.bbox(0) < b_min)
      plane_predecessor = idx;
  }

  const Eigen::Vector3d& _split_plane_normal = d.eigenvectors.col(2);
  IteratorType middle =
    split(begin, end, [&](const Eigen::Vector3d& p) -> bool { return (p - d.mean).dot(_split_plane_normal) < double(0); });

  if (true || level >= max_parallel_level) {
    MADNodeHandle{add_left(), storage_ptr}.build(
      vec, begin, middle, b_max, b_min, level + 1, max_parallel_level, idx, plane_predecessor);

    MADNodeHandle{add_right(), storage_ptr}.build(
      vec, middle, end, b_max, b_min, level + 1, max_parallel_level, idx, plane_predecessor);
  }
  // TODO to do async I should put a mutex on the storage. Let's see if it's needed hopefully not
  //  else {
  //   std::future<MADtree*> l =
  //     std::async(MADtree::makeSubtree, vec, begin, middle, b_max, b_min, level + 1, max_parallel_level, this,
  //     plane_predecessor);

  //   std::future<MADtree*> r =
  //     std::async(MADtree::makeSubtree, vec, middle, end, b_max, b_min, level + 1, max_parallel_level, this,
  //     plane_predecessor);
  //   left_  = l.get();
  //   right_ = r.get();
  // }
}

void MADtreeV::MADNodeHandle::apply_transform(const Eigen::Matrix3d& r, const Eigen::Vector3d& t) {
  auto& n = node();
  auto& d = n.data;

  d.mean         = r * d.mean + t;
  d.eigenvectors = r * d.eigenvectors;
  if (has_left())
    MADNodeHandle{n.left, storage_ptr}.apply_transform(r, t);
  if (has_right())
    MADNodeHandle{n.right, storage_ptr}.apply_transform(r, t);
}

const MADnode MADtreeV::MADNodeHandle::best_matching_leaf_fast(const Eigen::Vector3d& query) const {
  MADNodeHandle nh(*this);

  while (nh.has_left() || nh.has_right()) {
    // std::cerr << "idx:   " << nh.idx << "\n";
    // std::cerr << "left:  " << nh.node().left << "\n";
    // std::cerr << "right: " << nh.node().right << "\n";
    const Eigen::Vector3d& split_plane_normal = nh.node().data.eigenvectors.col(2);
    auto val                                  = ((query - nh.node().data.mean).dot(split_plane_normal));
    if (val < double(0)) {
      nh = MADNodeHandle{nh.node().left, storage_ptr};
    } else {
      nh = MADNodeHandle{nh.node().right, storage_ptr};
    }
  }

  return nh.node().data;
}

void MADtreeV::MADNodeHandle::get_leaves(std::back_insert_iterator<LeafList> it) {
  auto& n = node();
  if (!has_left() && !has_right()) {
    ++it = &n.data;
    return;
  }
  if (has_left())
    MADNodeHandle{n.left, storage_ptr}.get_leaves(it);
  if (has_right())
    MADNodeHandle{n.right, storage_ptr}.get_leaves(it);
}
