# Magnum IO Developer Environment container recipe

Stage0 += comment('GENERATED FILE, DO NOT EDIT')

Stage0 += baseimage(image='nvcr.io/nvidia/cuda:11.2.2-devel-ubuntu20.04')

Stage0 += nsight_systems(cli=True, version='2021.1.1')
Stage0 += mlnx_ofed(version='5.2-2.2.0.0', oslabel='ubuntu20.04')
Stage0 += gdrcopy(ldconfig=True, version='2.2')
Stage0 += ucx(version='1.10.0', cuda=True,
              gdrcopy='/usr/local/gdrcopy', ldconfig=True,
              disable_static=True, enable_mt=True)
Stage0 += nvshmem(version='2.0.2-0')

Stage0 += comment('GDS 0.95')
Stage0 += apt_get(
    keys=['https://repo.download.nvidia.com/baseos/GPG-KEY-dgx-cosmos-support'],
    ospackages=['libcufile-11-2=0.95.0.94-1',
                'libcufile-dev-11-2=0.95.0.94-1',
                'gds-tools-11-2=0.95.0.94-1'],
    repositories=['deb https://repo.download.nvidia.com/baseos/ubuntu/focal/x86_64/ focal-updates preview'])

Stage0 += copy(src=['magnum-io.Dockerfile', 'third_party.txt', 'README.md'], dest='/')

Stage0 += environment(variables={'MAGNUM_IO_VERSION': '21.04'})

Stage0 += raw(docker='SHELL ["/bin/bash", "-c"]\n\
CMD ["/bin/bash" ]')
