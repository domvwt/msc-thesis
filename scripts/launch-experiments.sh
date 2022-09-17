pip install --user . --no-dependencies

run_until_success () {
  while true; do
    $@ && break
    echo "Failed, retrying..."
  done
}


OPTUNA_DB="sqlite:///data/optuna-03.db"
EXPERIMENT_TYPE="DESIGN"
TARGET_TRIALS=30

#python src/mscproject/experiment.py -m ALL

python src/mscproject/experiment.py \
  -m GraphSAGE \
  --db $OPTUNA_DB \
  -n $TARGET_TRIALS \
  -e $EXPERIMENT_TYPE

python src/mscproject/experiment.py \
  -m GCN \
  --db $OPTUNA_DB \
  -n $TARGET_TRIALS \
  -e $EXPERIMENT_TYPE


EXPERIMENT_TYPE="HYPERPARAMETERS"
TARGET_TRIALS=20

python src/mscproject/experiment.py \
  -m GCN \
  --db $OPTUNA_DB \
  -n $TARGET_TRIALS \
  -e $EXPERIMENT_TYPE \

python src/mscproject/experiment.py \
  -m GraphSAGE \
  --db $OPTUNA_DB \
  -n $TARGET_TRIALS \
  -e $EXPERIMENT_TYPE \


EXPERIMENT_TYPE="TRAIN_ONLY"
TARGET_TRIALS=10

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


exit 0 
