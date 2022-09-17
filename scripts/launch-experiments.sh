pip install --user . --no-dependencies

run_until_success () {
  while true; do
    $@ && break
    echo "Failed, retrying..."
  done
}

OPTUNA_DB="sqlite:///data/optuna-03.db"
EXPERIMENT_TYPE="DESIGN"
TARGET_TRIALS=200

#python src/mscproject/experiment.py -m ALL

python src/mscproject/experiment.py \
  -m GCN \
  --db $OPTUNA_DB \
  -n $TARGET_TRIALS \
  -e $EXPERIMENT_TYPE

python src/mscproject/experiment.py \
  -m GraphSAGE \
  --db $OPTUNA_DB \
  -n $TARGET_TRIALS \
  -e $EXPERIMENT_TYPE

# python src/mscproject/experiment.py -m GAT -db $OPTUNA_DB
# python src/mscproject/experiment.py -m HGT -db $OPTUNA_DB
# python src/mscproject/experiment.py -m HAN -db $OPTUNA_DB

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# run_until_success python src/mscproject/experiment.py -m GAT
# run_until_success python src/mscproject/experiment.py -m HAN

exit 0 
