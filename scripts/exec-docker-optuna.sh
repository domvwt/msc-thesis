echo "Launching optuna studies - vm will be terminated when process completes."

docker run -v ${HOME}/msc-thesis:/workspace --rm --init --runtime nvidia --gpus all mscproject:0.1 \
	"scripts/launch-optuna.sh"

sudo shutdown +2 "System shutdown triggered (execute 'sudo shutdown -c' to cancel)"

