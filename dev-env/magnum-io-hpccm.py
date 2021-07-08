# Magnum IO Developer Environment container recipe

Stage0 += comment('GENERATED FILE, DO NOT EDIT')

Stage0 += baseimage(image='nvcr.io/nvidia/cuda:11.4.0-devel-ubuntu20.04')
# GDS 1.0 is part of the CUDA base image

Stage0 += nsight_systems(cli=True, version='2021.2.1')
Stage0 += mlnx_ofed(version='5.3-1.0.0.1')
Stage0 += gdrcopy(ldconfig=True, version='2.2')
Stage0 += ucx(version='1.10.1', cuda=True,
              gdrcopy='/usr/local/gdrcopy', ldconfig=True,
              disable_static=True, enable_mt=True)
Stage0 += nvshmem(version='2.2.1') # See hack in instaler.sh for 2.2.1 artifact renaming
Stage0 += nccl(cuda='11.4', version='2.10.3-1')

Stage0 += copy(src=['magnum-io.Dockerfile', 'third_party.txt', 'README.md'], dest='/')

Stage0 += environment(variables={'MAGNUM_IO_VERSION': '21.07'})

Stage0 += raw(docker='SHELL ["/bin/bash", "-c"]\n\
CMD ["/bin/bash" ]')
