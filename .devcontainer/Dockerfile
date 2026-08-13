FROM nvcr.io/nvidia/pytorch:26.03-py3

# Install useful dev tools
RUN apt-get update && apt-get -y install sudo nano gdb time htop nvtop redis

# Install clang from the signed LLVM apt repository
ENV CLANG_VER=20
RUN apt update
RUN apt install -y lsb-release software-properties-common gnupg
# Fetch the LLVM apt signing key and verify its fingerprint before trusting it
RUN wget -qO /tmp/llvm-snapshot.gpg.key https://apt.llvm.org/llvm-snapshot.gpg.key && \
    gpg --dearmor -o /usr/share/keyrings/llvm-snapshot.gpg /tmp/llvm-snapshot.gpg.key && \
    test "$(gpg --show-keys --with-colons /usr/share/keyrings/llvm-snapshot.gpg | awk -F: '$1 == "pub" { getline; print $10 }')" = \
         "6084F3CF814B57C1CF12EFD515CF4D18AF4F7421" && \
    rm /tmp/llvm-snapshot.gpg.key
RUN echo "deb [signed-by=/usr/share/keyrings/llvm-snapshot.gpg] http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-${CLANG_VER} main" > /etc/apt/sources.list.d/llvm.list
RUN apt update
RUN apt install -y clang-${CLANG_VER} lldb-${CLANG_VER} lld-${CLANG_VER} clangd-${CLANG_VER} \
    clang-tools-${CLANG_VER} clang-format-${CLANG_VER} clang-tidy-${CLANG_VER}

# Create user matching the host user
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Relax image constraint for torch
RUN sed -i '/^torch==/c\torch>=2.10' /etc/pip/constraint.txt

USER $USERNAME

# Keep container running
CMD ["sleep", "infinity"]
