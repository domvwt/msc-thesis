pip install --user . --no-dependencies

run_until_success () {
  while true; do
    $@ && break
    echo "Failed, retrying..."
  done
}


OPTUNA_DB="sqlite:///data/optuna-03.db"
STUDY_TYPE="ARCHITECTURE"
TARGET_TRIALS=30

#python src/mscproject/experiment.py -m ALL

# python src/mscproject/experiment.py \
#   -m GraphSAGE \
#   --db $OPTUNA_DB \
#   -n $TARGET_TRIALS \
#   -s $STUDY_TYPE

python src/mscproject/experiment.py \
  -m GCN \
  --db $OPTUNA_DB \
  -n $TARGET_TRIALS \
  -s $STUDY_TYPE


TARGET_TRIALS=60

python src/mscproject/experiment.py \
  -m GraphSAGE \
  --db $OPTUNA_DB \
  -n $TARGET_TRIALS \
  -s $STUDY_TYPE


STUDY_TYPE="REGULARISATION"
TARGET_TRIALS=20

python src/mscproject/experiment.py \
  -m GCN \
  --db $OPTUNA_DB \
  -n $TARGET_TRIALS \
  -s $STUDY_TYPE \

python src/mscproject/experiment.py \
  -m GraphSAGE \
  --db $OPTUNA_DB \
  -n $TARGET_TRIALS \
  -s $STUDY_TYPE \


STUDY_TYPE="WEIGHTS"
TARGET_TRIALS=10

python src/mscproject/experiment.py \
  -m GCN \
  --db $OPTUNA_DB \
  -n $TARGET_TRIALS \
  -s $STUDY_TYPE

python src/mscproject/experiment.py \
  -m GraphSAGE \
  --db $OPTUNA_DB \
  -n $TARGET_TRIALS \
  -s $STUDY_TYPE


exit 0 
