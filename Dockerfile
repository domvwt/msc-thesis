FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

LABEL org.opencontainers.image.authors = "Dominic Thorn"
LABEL org.opencontainers.image.base.name="pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel"

# Set conda executable.
ENV CONDA=/opt/conda/bin/conda

# Update conda.
RUN ${CONDA} config --add channels pytorch \
 && ${CONDA} config --add channels pyg \
 && ${CONDA} config --append channels conda-forge \
 && ${CONDA} update -n base conda -y \
 && ${CONDA} install -n base conda-libmamba-solver -y \
 && ${CONDA} config --set experimental_solver libmamba

# Install pytorch geometric.
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html \
 && pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html \
 && pip install torch-geometric \
 && python -m pip install torch-geometric==2.1.* torchmetrics==0.9.3

# Update .bashrc.
RUN ${CONDA} init \
 && echo "conda activate base" >> ~/.bashrc

# Install the project.
COPY ./requirements.txt ./
RUN python -m pip install -r requirements.txt \
 && pip install torchmetrics --no-dependencies

# Set default command.
CMD ["/bin/bash"]
