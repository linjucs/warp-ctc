
## Install

1. Follow top-level [README.md](warp-ctc/blob/master/README.md#compilation) and
   install warp-ctc.
2. Install [PyTorch](https://pytorch.org/)
3. In this directory run `python build.py`. 

### Troubleshooting

If you see something like `fatal error: cuda.h: No such file or directory`,
find the path to your cuda header files and pass that as an optional argument
to `build.py`.

For example:

```python build.py /usr/local/cuda/include```

## Test

In this directory run `python test.py`.
