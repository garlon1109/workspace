FROM nvcr.io/nvidia/tensorrt:21.05-py3

ENV DEBIAN_FRONTEND=noninteractive

### change the system source for installing libs
ARG USE_MIRROR
RUN if [ ! -z "${USE_MIRROR}" ] ; \
    then \
        sed -i s/archive.ubuntu.com/${USE_MIRROR}/g /etc/apt/sources.list ; \
        sed -i s/security.ubuntu.com/${USE_MIRROR}/g /etc/apt/sources.list ; \
        pip config set global.index-url https://${USE_MIRROR}/pypi/web/simple ; \
        echo "Use ${USE_MIRROR} for installing libs" ; \
    else \
        echo "Keep the download source unchanged" ; \
    fi

RUN apt-get update \
    && apt-get install -y ca-certificates curl sudo --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Build dependencies
ENV TZ=Etc/UTC
RUN apt-get update \
    && apt-get install -y cmake clang llvm-dev libclang-dev libjsoncpp-dev --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN rm /usr/local/bin/cmake && sudo ln -s /usr/bin/cmake /usr/local/bin/cmake

USER $USERNAME

WORKDIR /home/$USERNAME

# Install Rust
ARG USE_RUST
RUN if [ ! -z ${USE_RUST} ]; then \
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > ./rustup.sh \
      && chmod +x ./rustup.sh \
      && ./rustup.sh -y --default-toolchain nightly-2022-11-28 \
      && rm -r ./rustup.sh; \
    fi

ARG USE_PYTORCH
RUN if [ ! -z ${USE_PYTORCH} ] ; then \
      python3 -m pip install --upgrade pip \
      && python3 -m pip install torch torchvision \
      && python3 -m pip cache purge ; \
    fi
