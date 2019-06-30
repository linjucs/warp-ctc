import os
import sys
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup
this_file = os.path.abspath(__file__)
warp_root = os.path.dirname(os.path.dirname(this_file))

sources = ['src/warp_ctc.c']
headers = ['src/warp_ctc.h']
include_dirs = [os.path.join(warp_root, 'include')]
libraries = ['warpctc']
library_dirs = [os.path.join(warp_root, 'build')]
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/warp_ctc_cuda.c']
    headers += ['src/warp_ctc_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

    if len(sys.argv) > 1:
        include_dirs += [sys.argv[1]]

# ffi = CUDAExtension(
#     '_ext.ctc',
#     headers=headers,
#     sources=sources,
#     include_dirs=include_dirs,
#     relative_to=__file__,
#     define_macros=defines,
#     with_cuda=with_cuda,
#     libraries=libraries,
#     library_dirs=library_dirs
# )

# ffi.build()
setup(
        name='extension',
        ext_modules=[
            CUDAExtension(
                name='extension',
               
                sources=sources,
              
                extra_compile_args={'cxx': ['-g'],
                                        'nvcc': ['-O2']}),
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
