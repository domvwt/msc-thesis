pip install --user . --no-dependencies

run_until_success () {
  while true; do
    $@ && break
    echo "Failed, retrying..."
  done
}

# DEBUG
#TARGET_TRIALS=2

OPTUNA_DB="sqlite:///data/optuna-04.db"
STUDY_TYPE="ARCHITECTURE"
TARGET_TRIALS=100

#python src/mscproject/experiment.py -m ALL

python src/mscproject/experiment.py \
  -m KGNN \
  --db $OPTUNA_DB \
  -n $TARGET_TRIALS \
  -s $STUDY_TYPE \
  #--overwrite

python src/mscproject/experiment.py \
  -m GraphSAGE \
  --db $OPTUNA_DB \
  -n $TARGET_TRIALS \
  -s $STUDY_TYPE \
  #--overwrite


STUDY_TYPE="REGULARISATION"
BASE_STUDY_TYPE="ARCHITECTURE"
TARGET_TRIALS=30

python src/mscproject/experiment.py \
  -m KGNN \
  --db $OPTUNA_DB \
  -n $TARGET_TRIALS \
  -s $STUDY_TYPE \
  -b $BASE_STUDY_TYPE \
  #--overwrite

python src/mscproject/experiment.py \
  -m GraphSAGE \
  --db $OPTUNA_DB \
  -n $TARGET_TRIALS \
  -s $STUDY_TYPE \
  -b $BASE_STUDY_TYPE \
  #--overwrite


STUDY_TYPE="WEIGHTS"
BASE_STUDY_TYPE="ARCHITECTURE"
TARGET_TRIALS=50

python src/mscproject/experiment.py \
  -m KGNN \
  --db $OPTUNA_DB \
  -n $TARGET_TRIALS \
  -s $STUDY_TYPE \
  -b $BASE_STUDY_TYPE \
  #--overwrite

python src/mscproject/experiment.py \
  -m GraphSAGE \
  --db $OPTUNA_DB \
  -n $TARGET_TRIALS \
  -s $STUDY_TYPE \
  -b $BASE_STUDY_TYPE \
  #--overwrite


exit 0 
