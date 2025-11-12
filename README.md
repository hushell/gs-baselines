# GS baselines

## :package: Installation

### uv environment (a newer venv)
```
uv venv cuda124 --python 3.10
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### pip packages
```
uv pip install -r requirements.txt
uv pip install submodules/diff-surfel-rasterization --no-build-isolation
uv pip install submodules/diff-gaussian-rasterization --no-build-isolation
uv pip install submodules/simple-knn --no-build-isolation
```

### Additional package for OctFormer
```
git clone https://github.com/octree-nn/dwconv.git && cd dwconv && uv pip install -e . --no-build-isolation
```


