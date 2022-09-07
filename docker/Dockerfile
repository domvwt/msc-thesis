FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

LABEL org.opencontainers.image.version = "2.0.4"
LABEL org.opencontainers.image.authors = "Dominic Thorn"
LABEL org.opencontainers.image.source = "https://github.com/pyg-team/pytorch_geometric"
LABEL org.opencontainers.image.licenses = "MIT"
LABEL org.opencontainers.image.base.name="pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel"

# Set conda executable.
ENV CONDA=/opt/conda/bin/conda

# Update conda.
RUN ${CONDA} config --add channels pytorch \
 && ${CONDA} config --add channels pyg \
 && ${CONDA} update -n base conda -y \
 && ${CONDA} install -n base conda-libmamba-solver -y \
 && ${CONDA} config --set experimental_solver libmamba \
 && ${CONDA} config --append channels conda-forge

# Install pytorch geometric.
RUN ${CONDA} install -n base pyg==2.0.4 tensorboardx -y

# Install graphgym.
RUN git clone https://github.com/pyg-team/pytorch_geometric.git \
 && cd pytorch_geometric \
 && git checkout tags/2.0.4  \
 && cd ..  \
 && mv pytorch_geometric/graphgym .  \
 && rm -rf pytorch_geometric

# Update .bashrc.
RUN ${CONDA} init \
 && echo "conda activate base" >> ~/.bashrc

# Set default command.
CMD ["bash" "--login"]
