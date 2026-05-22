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

#pragma once

#include "mad_icp.h"
#include <tools/lie_algebra.h>
#include <tools/mad_tree.h>

#include <Eigen/Core>
#include <numeric>
#include <sys/time.h>
#include <vector>

class Localizer {
public:
  Localizer(double sensor_hz,
            bool deskew,
            double b_max,
            double rho_ker,
            double b_min,
            double b_ratio,
            int num_keyframes,
            int num_threads,
            int max_icp_its);
  ~Localizer();

  // Load pre-built map. Trees are NOT owned; caller keeps them alive.
  void loadMap(const std::vector<MADtree*>& trees,
               const std::vector<Eigen::Vector3d>& positions);

  // T_init: 4×4 initial guess (from bundlee interpolation).
  void compute(const double& stamp, ContainerType cloud, const Eigen::Matrix4d& T_init);

  // clang-format off
  const Eigen::Matrix4d currentPose()  const { return frame_to_map_.matrix(); }
  const bool isInitialized()           const { return is_initialized_; }
  const size_t currentID()             const { return seq_; }
  const bool isMapUpdated()            const { return is_map_updated_; }
  // clang-format on

  const ContainerType currentLeaves();
  const ContainerType modelLeaves();

protected:
  void deskew(const ContainerTypePtr& cloud,
              const Eigen::Isometry3d& T_prev,
              const Eigen::Isometry3d& T_now);
  void selectActiveKeyframes(const Eigen::Vector3d& position);

  MADicp icp_;

  // full map (not owned)
  std::vector<MADtree*>        map_trees_;
  std::vector<Eigen::Vector3d> map_positions_;

  // currently active subset
  std::vector<MADtree*> active_trees_;
  LeafList              active_leaves_;

  MADtree* current_tree_;
  LeafList current_leaves_;

  Eigen::Isometry3d              frame_to_map_;
  std::vector<Eigen::Isometry3d> trajectory_;

  bool   deskew_, is_initialized_, is_map_updated_;
  int    num_keyframes_, num_threads_, max_parallel_levels_, max_icp_its_;
  double sensor_hz_, b_max_, b_min_;
  size_t seq_;
};
