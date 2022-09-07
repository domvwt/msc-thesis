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
RUN ${CONDA} install -n base pyg==2.0.4

# Update .bashrc.
RUN ${CONDA} init \
 && echo "conda activate base" >> ~/.bashrc

# Set default command.
CMD ["bash" "--login"]
