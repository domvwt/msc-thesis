pip install --user . --no-dependencies

run_until_success () {
  while true; do
    $@ && break
    echo "Failed, retrying..."
  done
}

#python src/mscproject/experiment.py -m ALL  # COMPLETED
python src/mscproject/experiment.py -m GCN  # COMPLETED
python src/mscproject/experiment.py -m GraphSAGE  # COMPLETED
#python src/mscproject/experiment.py -m GAT  # ONGOING
python src/mscproject/experiment.py -m HGT  # COMPLETED
#python src/mscproject/experiment.py -m HAN  # COMPLETED

#python src/mscproject/experiment.py -m GCN --overwrite

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

run_until_success python src/mscproject/experiment.py -m GAT
run_until_success python src/mscproject/experiment.py -m HAN

exit 0 
