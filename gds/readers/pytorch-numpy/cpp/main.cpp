// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "numpy_reader.h"

int main(int argc, char* argv[]){

  std::string filename = "../tests/arr_linear.npy";
  NumpyReader npr(false, -1);

  //init file
  npr.Initile(filename);
  npr.ParseFile(filename);
  
  return 0;
}
