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

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <cstring>

// mad_tree_wrapper.h already pulls in eigen_stl_bindings.h
#include "tools/mad_tree_wrapper.h"
#include "odometry/localizer.h"

namespace py = pybind11;

PYBIND11_MODULE(pylocalizer, m) {

  py::class_<Localizer>(m, "Localizer")
    .def(py::init<double, bool, double, double, double, double, int, int, int>(),
         py::arg("sensor_hz"),
         py::arg("deskew"),
         py::arg("b_max"),
         py::arg("rho_ker"),
         py::arg("b_min"),
         py::arg("b_ratio"),
         py::arg("num_keyframes"),
         py::arg("num_threads"),
         py::arg("max_icp_its") = 15)

    // loadMap: accept list of MADtreeWrapper* + (N,3) numpy positions
    .def("loadMap",
         [](Localizer& loc,
            const std::vector<MADtreeWrapper*>& wrappers,
            py::array_t<double, py::array::c_style | py::array::forcecast> positions) {
           auto buf = positions.request();
           if (buf.ndim != 2 || buf.shape[1] != 3)
             throw std::runtime_error("positions must be (N,3)");
           const int n = buf.shape[0];
           if (n != (int)wrappers.size())
             throw std::runtime_error("trees and positions must have the same length");

           std::vector<MADtree*> trees;
           trees.reserve(n);
           for (MADtreeWrapper* w : wrappers) trees.push_back(w->getTree());

           std::vector<Eigen::Vector3d> pos(n);
           const double* data = static_cast<const double*>(buf.ptr);
           for (int i = 0; i < n; ++i)
             pos[i] = Eigen::Vector3d(data[3 * i], data[3 * i + 1], data[3 * i + 2]);

           loc.loadMap(trees, pos);
         },
         py::arg("trees"), py::arg("positions"))

    // compute: accept raw numpy (N,3) — single memcpy into ContainerType, no Python overhead
    .def("compute",
         [](Localizer& loc,
            double stamp,
            py::array_t<double, py::array::c_style | py::array::forcecast> pts,
            const Eigen::Matrix4d& T_init) {
           auto buf = pts.request();
           if (buf.ndim != 2 || buf.shape[1] != 3)
             throw std::runtime_error("pts must be (N,3)");
           const int n = buf.shape[0];
           ContainerType cloud(n);
           // Eigen::Vector3d is 3 doubles with no padding — layout matches numpy (N,3) float64
           std::memcpy(cloud.data(), buf.ptr, n * 3 * sizeof(double));
           loc.compute(stamp, std::move(cloud), T_init);
         },
         py::arg("stamp"), py::arg("pts"), py::arg("T_init"))

    .def("currentPose",    &Localizer::currentPose)
    .def("isInitialized",  &Localizer::isInitialized)
    .def("currentID",      &Localizer::currentID)
    .def("isMapUpdated",   &Localizer::isMapUpdated)
    .def("currentLeaves",  &Localizer::currentLeaves)
    .def("modelLeaves",    &Localizer::modelLeaves);
}
