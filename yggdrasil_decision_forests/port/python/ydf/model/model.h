/*
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_DATASET_MODEL_H_
#define YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_DATASET_MODEL_H_

#include <pybind11/pybind11.h>

#include <memory>

#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "ydf/model/model_wrapper.h"

namespace yggdrasil_decision_forests::port::python {

std::unique_ptr<GenericCCModel> CreateCCModel(
    std::unique_ptr<model::AbstractModel> model_ptr);

void init_model(pybind11::module_ &m);

}  // namespace yggdrasil_decision_forests::port::python

#endif  // YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_DATASET_MODEL_H_
