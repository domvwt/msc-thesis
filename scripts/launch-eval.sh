pip install --user . --no-dependencies
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
#export CUDA_VISIBLE_DEVICES=""
python notebooks/16-pyg-evaluation.py
exit 0 
