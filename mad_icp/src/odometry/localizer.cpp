// Copyright 2024 R(obots) V(ision) and P(erception) group
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "localizer.h"
#include <tools/constants.h>
#include <cmath>

Localizer::Localizer(double sensor_hz,
                     bool deskew,
                     double b_max,
                     double rho_ker,
                     double b_min,
                     double b_ratio,
                     int num_keyframes,
                     int num_threads,
                     int max_icp_its) :
  sensor_hz_(sensor_hz),
  deskew_(deskew),
  b_max_(b_max),
  b_min_(b_min),
  num_keyframes_(num_keyframes),
  num_threads_(num_threads),
  max_icp_its_(max_icp_its),
  icp_(b_max, rho_ker, b_ratio, num_threads),
  current_tree_(nullptr) {
  frame_to_map_.setIdentity();
  seq_             = 0;
  is_initialized_  = false;
  is_map_updated_  = false;
  max_parallel_levels_ = static_cast<int>(std::log2(num_threads));
  omp_set_num_threads(num_threads);
}

Localizer::~Localizer() {
  if (current_tree_) delete current_tree_;
}

void Localizer::loadMap(const std::vector<MADtree*>& trees,
                        const std::vector<Eigen::Vector3d>& positions) {
  map_trees_     = trees;
  map_positions_ = positions;
}

void Localizer::selectActiveKeyframes(const Eigen::Vector3d& position) {
  const size_t n = std::min((size_t)num_keyframes_, map_trees_.size());

  std::vector<size_t> idx(map_trees_.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + n, idx.end(),
    [&](size_t a, size_t b) {
      return (map_positions_[a] - position).squaredNorm() <
             (map_positions_[b] - position).squaredNorm();
    });

  // check if the active set actually changed
  bool changed = (active_trees_.size() != n);
  if (!changed) {
    for (size_t i = 0; i < n; ++i) {
      if (active_trees_[i] != map_trees_[idx[i]]) { changed = true; break; }
    }
  }

  if (!changed) {
    is_map_updated_ = false;
    return;
  }

  active_trees_.clear();
  active_leaves_.clear();
  for (size_t i = 0; i < n; ++i) {
    active_trees_.push_back(map_trees_[idx[i]]);
    map_trees_[idx[i]]->getLeafs(std::back_insert_iterator<LeafList>(active_leaves_));
  }
  is_map_updated_ = true;
}

void Localizer::deskew(const ContainerTypePtr& cloud,
                       const Eigen::Isometry3d& T_prev,
                       const Eigen::Isometry3d& T_now) {
  const double ts = 1. / sensor_hz_;

  Vector6d naive_vel;
  Eigen::Isometry3d T_now_to_prev = T_prev.inverse() * T_now;
  naive_vel.head(3)               = T_now_to_prev.translation();
  naive_vel.tail(3)               = logMapSO3(T_now_to_prev.linear());
  naive_vel                       = naive_vel / ts;

  using AzimuthPair = std::pair<double, Eigen::Vector3d>;
  std::vector<AzimuthPair> sorted(cloud->size());
  for (size_t i = 0; i < sorted.size(); ++i)
    sorted[i] = {std::atan2(cloud->at(i).y(), cloud->at(i).x()), cloud->at(i)};

  std::sort(sorted.begin(), sorted.end(),
    [](const AzimuthPair& a, const AzimuthPair& b) { return a.first < b.first; });

  const double resolution = 2 * M_PI / double(CHUNKS);
  const double delta      = ts / double(CHUNKS - 1);
  double t                = -ts;

  Vector6d delta_s = naive_vel * t;
  Eigen::Isometry3d meas_pose_to_robot = Eigen::Isometry3d::Identity();
  meas_pose_to_robot.linear()      = expMapSO3(delta_s.tail(3));
  meas_pose_to_robot.translation() = delta_s.head(3);

  double angle = M_PI - resolution;
  for (int i = (int)sorted.size() - 1; i >= 0; --i) {
    if (sorted[i].first < angle) {
      angle -= resolution;
      t += delta;
      delta_s                          = naive_vel * t;
      meas_pose_to_robot               = Eigen::Isometry3d::Identity();
      meas_pose_to_robot.linear()      = expMapSO3(delta_s.tail(3));
      meas_pose_to_robot.translation() = delta_s.head(3);
    }
    (*cloud)[i] = meas_pose_to_robot * sorted[i].second;
  }
}

void Localizer::compute(const double& /*stamp*/, ContainerType cloud_mem,
                        const Eigen::Matrix4d& T_init_mat) {
  ContainerType* cloud = &cloud_mem;

  if (deskew_ && trajectory_.size() > 1)
    deskew(cloud, trajectory_[trajectory_.size() - 2], trajectory_[trajectory_.size() - 1]);

  if (current_tree_) delete current_tree_;
  current_tree_ = new MADtree(
    cloud, cloud->begin(), cloud->end(), b_max_, b_min_, 0, max_parallel_levels_, nullptr, nullptr);
  current_leaves_.clear();
  current_tree_->getLeafs(std::back_insert_iterator<LeafList>(current_leaves_));

  Eigen::Isometry3d T_init;
  T_init.linear()      = T_init_mat.block<3, 3>(0, 0);
  T_init.translation() = T_init_mat.block<3, 1>(0, 3);

  selectActiveKeyframes(T_init.translation());

  icp_.setMoving(current_leaves_);
  icp_.init(T_init);

  for (int it = 0; it < max_icp_its_; ++it) {
    if (it == max_icp_its_ - 1)
      for (MADtree* l : current_leaves_) l->matched_ = false;

    icp_.resetAdders();

#pragma omp parallel for
    for (size_t k = 0; k < active_trees_.size(); ++k)
      icp_.update(active_trees_[k]);

#pragma omp barrier

    icp_.updateState();
  }

  frame_to_map_ = icp_.X_;
  trajectory_.push_back(frame_to_map_);
  is_initialized_ = true;
  seq_++;
}

const ContainerType Localizer::currentLeaves() {
  ContainerType out;
  out.reserve(current_leaves_.size());
  for (MADtree* l : current_leaves_) out.push_back(l->mean_);
  return out;
}

const ContainerType Localizer::modelLeaves() {
  ContainerType out;
  out.reserve(active_leaves_.size());
  for (MADtree* l : active_leaves_) out.push_back(l->mean_);
  return out;
}
