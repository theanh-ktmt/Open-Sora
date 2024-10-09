# Quick Guidance

## Environment
**Running environment**
- CUDA: 12.3
- cuDNN: 8.9.3
- TensorRT: 10.4.0
- PyCUDA: 2024.1.2

**Check library version**
- Check Nvidia driver
- Check `CUDA` version
```
nvcc  --version
```
- Check `cuDNN` version
```
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

# The result will be, corresponding to version 8.9.3
#define CUDNN_MAJOR 8
#define CUDNN_MINOR 9
#define CUDNN_PATCHLEVEL 3
```
- Check `TensorRT`
```
pip show tensorrt

# If not available, installing using this command
pip install tensorrt
```
- Check `PyCUDA`:
```
pip show pycuda

# If not available, installing using this command
pip install pycuda
```
