docker run -v ${HOME}/msc-thesis:/workspace --rm --init --runtime nvidia --gpus all mscproject:0.1 \
	"scripts/launch-eval.sh"
