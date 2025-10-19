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

#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <odometry/mad_icp.h>
#include <random>
#include <tools/mad_tree.h>
#include <tools/mad_tree_old.h>

Eigen::Matrix3d rpyToRotation(double roll, double pitch, double yaw) {
  Eigen::AngleAxisd Rz(yaw, Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd Ry(pitch, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd Rx(roll, Eigen::Vector3d::UnitX());
  return (Rz * Ry * Rx).toRotationMatrix(); // ZYX convention
}

// b_max : 0.2 # [m] max size of kd leaves
// b_min : 0.1 # [m] when a node is flatten than this param, propagate normal
// b_ratio : 0.02 # the increase factor of search radius needed in data association
// p_th : 0.8 # [%] ensuring an update when the curr point cloud is registered less than this param
// rho_ker : 0.1 # huber threshold in mad-icp
// n : 10 # the number of last poses to smooth velocity

int main(int argc, char* argv[]) {
  const int N = 150000; // number of vectors
  std::vector<Eigen::Vector3d> points;
  points.reserve(N);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(0.0, 1.0);

  for (int i = 0; i < N; ++i) {
    Eigen::Vector3d v(dist(gen), dist(gen), dist(gen));
    points.push_back(100 * v);
  }

  auto start2 = std::chrono::high_resolution_clock::now();
  // If tree2 builds in constructor, skip this, otherwise:
  MADtree tree2(&points, points.begin(), points.end(), 0.2, 0.1, 0, 4, nullptr, nullptr);
  auto end2                              = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed2 = end2 - start2;
  std::cout << "MADtree build took: " << elapsed2.count() << " seconds\n";

  // Profile MADtreeV
  MADtreeV tree1;
  auto start1                            = std::chrono::high_resolution_clock::now();
  auto root1                             = tree1.build(&points, points.begin(), points.end(), 0.2, 0.1, 4);
  auto end1                              = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed1 = end1 - start1;
  std::cout << "MADtreeV build took: " << elapsed1.count() << " seconds\n";

  return 0;
}
