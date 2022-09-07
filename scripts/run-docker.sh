docker run -it --rm --init --runtime nvidia --gpus all mscproject:latest /bin/bash \
    -v /home/projects/msc-thesis/:mscproject/ \