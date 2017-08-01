import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['src/warp_ctc.c']
headers = ['src/warp_ctc.h']
include_dirs = ['/afs/cs.stanford.edu/u/awni/scr/warp-ctc/include/']
libraries = ['warpctc']
library_dirs = ['/afs/cs.stanford.edu/u/awni/scr/warp-ctc/build/']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/warp_ctc_cuda.c']
    headers += ['src/warp_ctc_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True


ffi = create_extension(
    '_ext.ctc',
    headers=headers,
    sources=sources,
    include_dirs=include_dirs,
    relative_to=__file__,
    define_macros=defines,
    with_cuda=with_cuda,
    libraries=libraries,
    library_dirs=library_dirs
)

if __name__ == '__main__':
    ffi.build()
