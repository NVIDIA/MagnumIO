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

void NumpyReader::EnableP2P(){
  if(!_p2p_enabled){
    cudaGetDeviceCount(&_num_gpu);

    //loop over all devices and enable p2p
    for (int k = 0; k < _num_gpu; k++) {
      cudaSetDevice(k);
      for (int j = k+1; j < _num_gpu; j++) {
        int access = 0;
        cudaDeviceCanAccessPeer(&access, k, j);
        if (access) {
          cudaDeviceEnablePeerAccess(j, 0 );
          cudaSetDevice(j);
          cudaDeviceEnablePeerAccess(k, 0 );
          cudaSetDevice(k);
        }
      }
    }
    _p2p_enabled = true;
  }
}

void NumpyReader::DisableP2P(){
  if(_p2p_enabled){
    for (int k = 0; k < _num_gpu; k++) {
      cudaSetDevice(k);
      for (int j = k+1; j < _num_gpu; j++) {
        int access = 0;
        cudaDeviceCanAccessPeer(&access, k, j);
        if (access) {
          cudaDeviceDisablePeerAccess(j);
          cudaSetDevice(j);
          cudaDeviceDisablePeerAccess(k);
          cudaSetDevice(k);
        }
      }
    }
    _p2p_enabled = false;
  }
}

void NumpyReader::ParseFile(const std::string& filename){
  
  //check out https://numpy.org/neps/nep-0001-npy-format.html for format details
  
  //check if we need to reallocate
  bool realloc = false;

  //open file
  _fd = open(filename.c_str(), O_RDONLY);
  if(_fd == -1){
    handleIOError(_fd);
    std::rethrow_exception(_teptr);
  }
  
  //check if the file is actually a numpy file
  char* token = new char[10];
  int64_t nread = pread(_fd, token, 10, 0);
  if(nread != 10){
    //std::cout << std::string(token) << std::endl;
    throw std::runtime_error("NumpyReader: could not read header.");
  }
  //check if heqder is too short
  if(std::string(token).find("{") != std::string::npos){
    throw std::runtime_error("NumpyReader: header is corrupted.");
  }
  if(std::string(token).find("NUMPY") == std::string::npos){
    throw std::runtime_error("NumpyReader: not a numpy file.");
  }
  unsigned short header_len = 0;
  memcpy(&header_len, &token[8], 2);
  delete [] token;  
  
  //read header: the offset is a magic number
  _offset = (6+1+1+2);
  token = new char[header_len];
  nread = pread(_fd, token, header_len, _offset);
  
  //close file
  close(_fd);
  
  //parse the header now
  std::string header(token);
  delete [] token;
  _offset = (6+1+1+2+header_len);
  
  //extract dictionary info from header
  std::smatch header_match;
  if(!std::regex_search(header, header_match, _header_regex)){
    throw std::runtime_error("NumpyReader: cannot parse header, ill formatted or corrupt.");
  }
  
  //type
  std::string typestring = header_match[1].str();
  // < means LE, | means N/A, = means native. In all those cases, we can read
  _little_endian = (typestring[0] == '<' || typestring[0] == '|' || typestring[0] == '=' ? true : false);
  if(!_little_endian){
    throw std::runtime_error("NumpyReader: the specified file is in big endian. This is currently not supported.");
  }
  std::string tid = typestring.substr(1);
  //get type in a safe way
  std::map<std::string, torch::Dtype>::iterator typeit = _typemap.find(tid);
  if(typeit == _typemap.end()) throw std::runtime_error("NumpyReader: unsupported datatype.");
  if(typeit->second != _type){
    _type = typeit->second;
    realloc = true;
  }
  int64_t tsize = static_cast<int64_t>(stoi(typestring.substr(2)));
  if(tsize != _typesize){
    _typesize = tsize;
    realloc = true;
  }
  
  //data order
  bool order;
  if(header_match[2].str() == "False") order = false;
  else order = true;
  if(order != _fortran_order){
    _fortran_order = order;
    realloc = true;
  }
  
  //set sizes
  std::string shapestring = header_match[3].str();
  std::regex shape_regex{R"(,+)"}; // split on comma
  std::sregex_token_iterator it{shapestring.begin(), shapestring.end(), shape_regex, -1};
  std::vector<std::string> shapevec{it, {}};
  
  //if shapevec size is 1 and shapevec[0] is the empty string, it is a scalar/singleton (denoted as ()) and needs to be set to one:
  if( (shapevec.size() == 1) && (shapevec[0] == "") ) shapevec[0] = "1";

  //distinguish between a single sample and multiple samples in a file
  if(_split_axis){
    //if the array is in fortran order, error out (for the moment, could be supported later, but it is tricky because of the chunked reads)
    if(_fortran_order){
      throw std::runtime_error("NumpyReader: reading column-major arrays (Fortran order) is currently only supported if the split_axis option is false.");
    }
    //the first dim is the number of samples
    _numsample = static_cast<int64_t>(stoi(shapevec[0]));
    //remove the rest
    shapevec.erase(shapevec.begin());
    //we need to make sure shapevec is not empty:
    if(shapevec.empty()) shapevec.push_back("1");
  }
  else{
    _numsample = 1;
  }
  
  //now the shapevec should truly hold the tensor shape
  if(shapevec.size() != _shape.size()){
    _shape.clear();
    for(unsigned int i = 0; i < shapevec.size(); i++) _shape.push_back(static_cast<int64_t>(stoi(shapevec[i])));
    realloc = true;
  }  
  for(unsigned int i = 0; i < shapevec.size(); i++){
    int64_t elem = static_cast<int64_t>(stoi(shapevec[i]));
    if(_shape[i] != elem){
      _shape[i] = elem;
      realloc = true;
    }
  }
  
  //recompute strides
  computeStrides();
  
  if(realloc){
    allocate();
  }
}


//this is needed in order to recompute a stride if necessary
void NumpyReader::computeStrides(){
  _stride.resize(_shape.size());
  if(! _fortran_order){
    //row-major
    _stride[_stride.size() - 1] = 1;
    for(int i = static_cast<int>(_shape.size()) - 2; i >= 0; --i){
      _stride[i] = _shape[i+1] * _stride[i+1];
    }
  }
  else{
    //column-major
    _stride[0] = 1; 
    for(unsigned int i = 1; i < _shape.size(); ++i){
      _stride[i] = _shape[i-1] * _stride[i-1];
    }
  }
}


//print header info, good for debugging.
void NumpyReader::PrintHeaderInfo(){
	std::cout << "Fortran Order: " << (_fortran_order ? "Yes" : "No") << std::endl;
	std::cout << "Endianess: " << (_little_endian ? "little endian" : "big endian") << std::endl;
	std::cout << "Number of Samples: " << _numsample << std::endl;
	std::cout << "Number of Elements: " << _numelem << std::endl;
	std::cout << "Size per Element: " << _typesize << std::endl;
	std::cout << "Type id: " << _type << std::endl;
	if( _shape.size() > 0){
		std::cout << "Shape: (" << _shape[0];
		for(unsigned int i=1; i < _shape.size(); i++) std::cout << ", " << _shape[i];
		std::cout << ")" << std::endl;
	}
	else{
		std::cout << "Shape: (,)" << std::endl;
	}
  std::cout << "Stride: (" << _stride[0];
  for(unsigned int i=1; i < _stride.size(); i++) std::cout << ", " << _stride[i];
  std::cout << ")" << std::endl;
}


void NumpyReader::SetBatchsize(const unsigned int& batch_size){
  if( (batch_size == 0) || (batch_size > _numsample) ){
    throw std::out_of_range("NumpyReader: the batch size has to be a positive number and must not be bigger than the total number of samples.");
  }
  if( !_split_axis && batch_size > 1 ){
    throw std::out_of_range("NumpyReader: in order to use batching, you must have more than one sample per file. Otherwise batch externally.");
  }
  
  //do some sanity checks:
  //if this is non empty, we need to reallocate
  bool realloc = (_batchsize != batch_size);
  _batchsize = batch_size;
  
  if(realloc){
    allocate();
  }
}


void NumpyReader::allocate(){

  //cpu data
  if(_data != nullptr){
    delete [] _data;
    _data = nullptr;
  }

  //gpu data
  if(_ddata != nullptr){

    //deregister buffer
    deregisterBuffer();
    
    //free buffer
    cudaFree(_ddata);
    _ddata = nullptr;
  }
  
  //recompute sizes
  //array
  _numelem = 1;
  for(unsigned int i=0; i < _shape.size(); ++i) _numelem *= _shape[i];
  
  //allocate buffer for IO
  if( _device.is_cuda() ){
    
    //allocate buffer
    cudaMalloc((void**)&(_ddata), _batchsize * _numelem * _typesize);

    //register buffer
    registerBuffer();
  }
  else{
    _data = new unsigned char[_batchsize * _numelem * _typesize];
  }
  
  //we might need to enhance the shape-vector
  std::vector<int64_t> shape(_shape);
  if(_batchsize > 1){
    shape.insert(shape.begin(), _batchsize);
  }
  
  //we might need to enhance the stride-vector
  std::vector<int64_t> stride(_stride);
  if(_batchsize > 1){
    stride.insert(stride.begin(), _numelem);
  }
    
  //allocate buffer for IO
  auto options = torch::TensorOptions().dtype(_type).layout(torch::kStrided).requires_grad(false);
  if( _device.is_cuda() ){
    _sample = torch::from_blob(_ddata, shape, stride, options.device(_device));
  }
  else{
    _sample = torch::from_blob(_data, shape, stride, options);
  }
}

void NumpyReader::deregisterBuffer(){
  //de-register dev buffer
  for(size_t i = 0; i < _reg_buff.size(); ++i) {
    auto ptr = _reg_buff[i].first;
    cuFileBufDeregister(ptr);
  }
  _reg_buff.clear();
}

void NumpyReader::registerBuffer(){
  //register for cuFile: do registration per batch and then per intra-thread chunk
  size_t bsize = _numelem * _typesize;
  size_t nth = ((bsize / _num_intra_threads) > 1 ? _num_intra_threads : 1);
  size_t chunksize = std::ceil(bsize / nth);
  off_t offset = 0;
  for(int64_t b = 0; b < _batchsize; ++b){
    for(off_t t_off = 0; t_off < bsize; t_off += chunksize){
      // determine size of buffer and offset
      int64_t chunksize_eff = std::min(chunksize, bsize - t_off);
      auto ptr = static_cast<unsigned char*>(_ddata + offset + t_off);

      // register
      _cf_status = cuFileBufRegister(ptr, chunksize_eff, 0);
      if( _cf_status.err != CU_FILE_SUCCESS ){
        throw std::runtime_error("NumpyReader: failed to register device buffer.");
      }

      // memorize registered buffer data
      _reg_buff.push_back( std::make_pair(ptr, chunksize_eff) );
    }
    offset += bsize;
  }
}

void NumpyReader::InitFile(const std::string& filename){
  //open use DirectIO on the GPU
  if( ! _device.is_cuda() ){
    _fd = open(filename.c_str(), O_RDONLY);
  }
  else{
    _fd = open(filename.c_str(), O_RDONLY|O_DIRECT);
  }
  if(_fd == -1){
    handleIOError(_fd);
    std::rethrow_exception(_teptr);
  }
    
  if( _device.is_cuda() ){
    //set up cufile stuff
    _cf_descr.handle.fd = _fd;
    _cf_descr.fs_ops = nullptr;
    _cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    _cf_status = cuFileHandleRegister(&_cf_handle, &_cf_descr);
    if( _cf_status.err != CU_FILE_SUCCESS ){
      if( _cf_status.err == CU_FILE_IO_NOT_SUPPORTED ){
        throw std::runtime_error("NumpyReader: failed to register cufile handle. File system not supported.");
      }
      else{
        throw std::runtime_error("NumpyReader: failed to register cufile handle.");
      }
    }
  }
}


void NumpyReader::FinalizeFile(){
  if( _device.is_cuda() ){
    cuFileHandleDeregister(_cf_handle);
  }

  //close file
  close(_fd);
}

void NumpyReader::handleIOError(int64_t ret) {
  std::string errmsg;
  if (ret == -1) {
    std::string strbuff(256, '\0');
    int e = errno;
    errmsg = std::string(strerror_r(e, static_cast<char*>(&strbuff[0]), strbuff.size()));
    errmsg = "NumpyReader: " + errmsg;
  } else {
    errmsg = "NumpyReader: read failed with error " + std::to_string(-ret) + cufileop_status_error(static_cast<CUfileOpError>(-ret));
  }
  
  // throw exception
  try{
    throw std::ios_base::failure(errmsg);
  }
  catch(...){
    _teptr = std::current_exception();
  }
}


void NumpyReader::readChunk(int64_t dest_off, int64_t src_off, int64_t size){

  // set the device
  if (_device.is_cuda()) cudaSetDevice(_device.index());
  
  // handle offsets and read sizes
  int64_t nread;
  off_t file_off = src_off;
  off_t buff_off = dest_off;
  while (size > 0) {
    if (!_device.is_cuda()) {
      nread = pread(_fd, _data + buff_off, size, _offset + file_off);
    }
    else {
      nread = cuFileRead( _cf_handle, static_cast<void*>(_ddata + buff_off), size, static_cast<off_t>(_offset + file_off), 0 );
    }

    if (nread >= 0) {
      // worked well, continue
      size -= nread;
      file_off += nread;
      buff_off += nread;
    }
    else {
      handleIOError(nread);
      return;
    }
  }
}


void NumpyReader::readSample(int64_t dst_idx, int64_t src_idx){
  //private function used by get sample: read sample at index src into batch index dest:
  
  //rescale into reasonable range
  src_idx = (src_idx + _numsample) % _numsample;
  dst_idx = (dst_idx + _batchsize) % _batchsize;
  
  //size of data in file
  int64_t bsize = _numelem * _typesize;
  int64_t nth = ((bsize / _num_intra_threads) > 1 ? _num_intra_threads : 1);
  int64_t chunksize = std::ceil(bsize / nth);
  
  //dispatch loads
  for(int64_t t_off = 0; t_off < bsize; t_off += chunksize){
    int64_t chunksize_eff = std::min(chunksize, bsize - t_off);
    int64_t dst_off = (bsize * dst_idx) + t_off;
    int64_t src_off = (bsize * src_idx) + t_off;
    _thread_pool->AddWork([this, dst_off, src_off, chunksize_eff](int thread_id){
	readChunk(dst_off, src_off, chunksize_eff);
      });
  }
}


torch::Tensor NumpyReader::getSample(int64_t idx){
  
  //do sanity check
  if( _batchsize > 1 ){
    throw std::runtime_error("NumpyReader: please use getBatch to load a batch if batch-size > 1." );
  }
  
  //read one sample from id idx into batch 0
  readSample(0, idx);
  _thread_pool->RunAll(true);
  
  //check for exceptions
  if(_teptr){
    std::rethrow_exception(_teptr);
  }
  
  return _sample.clone();
}


//load a full batch with numbers
torch::Tensor NumpyReader::getBatch(const std::vector<int64_t>& indices){
  
  if( static_cast<int64_t>(indices.size()) != _batchsize ){
    throw std::runtime_error("NumpyReader: please make sure that the number of items matches the batchsize." );
  }
  
  //dispatch loads
  for(unsigned int idx = 0; idx < _batchsize; idx++){
    readSample(idx, indices[idx]);
  }
  _thread_pool->RunAll(true);
  
  //check for exceptions
  if(_teptr){
    std::rethrow_exception(_teptr);
  }
  
  return _sample.clone();
}


//python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  //class
  py::class_<NumpyReader> npr(m, "numpy_reader");
  
  //constructor
  npr.def(py::init<const bool&, const int&>(), py::arg("split_axis") = false, py::arg("device") = -1 );

  //constant properties
  npr.def_property_readonly("num_samples", [](const NumpyReader& numpyr) { return numpyr.getNumSamples(); });
  npr.def_property_readonly("shape", [](const NumpyReader& numpyr) { return numpyr.getShape(); });
  npr.def_property_readonly("strides", [](const NumpyReader& numpyr) { return numpyr.getStrides(); });

  //properties
  npr.def_property("num_inter_threads",
		   [](const NumpyReader& numpyr) { return numpyr.GetInterThreads(); }, 
		   [](NumpyReader& numpyr, const unsigned int& val) -> void { numpyr.SetInterThreads(val); });
  
  npr.def_property("num_intra_threads",
		   [](const NumpyReader& numpyr) { return numpyr.GetIntraThreads(); },
		   [](NumpyReader& numpyr, const unsigned int& val) -> void { numpyr.SetIntraThreads(val); });

  //accessors
  npr.def("set_batchsize", &NumpyReader::SetBatchsize, py::arg("batch_size"));
  
  //class functions
  npr.def("parse", &NumpyReader::ParseFile, py::arg("filename") );
  npr.def("init_file", &NumpyReader::InitFile, py::arg("filename") );
  npr.def("finalize_file", &NumpyReader::FinalizeFile);
  npr.def("print_file_info", &NumpyReader::PrintHeaderInfo);
  npr.def("enable_p2p", &NumpyReader::EnableP2P);
  npr.def("disable_p2p", &NumpyReader::DisableP2P);
  
  //loader
  npr.def("get_sample", &NumpyReader::getSample, py::arg("element_id"), py::call_guard<py::gil_scoped_release>());
  npr.def("get_batch", &NumpyReader::getBatch, py::arg("element_ids"), py::call_guard<py::gil_scoped_release>());
}
