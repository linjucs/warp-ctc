import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['src/warp_ctc.c']
headers = ['src/warp_ctc.h']
include_dirs = ['/afs/cs.stanford.edu/u/awni/scr/warp-ctc/include/']
libraries = ['warpctc']
library_dirs = ['/afs/cs.stanford.edu/u/awni/scr/warp-ctc/build/']
with_cuda = False

ffi = create_extension(
    '_ext.ctc',
    headers=headers,
    sources=sources,
    include_dirs=include_dirs,
    relative_to=__file__,
    with_cuda=with_cuda,
    libraries=libraries,
    library_dirs=library_dirs
)

if __name__ == '__main__':
    ffi.build()
