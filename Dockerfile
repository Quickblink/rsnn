FROM nvcr.io/nvidia/pytorch:20.01-py3
RUN apt-get update
RUN apt-get install -y mesa-utils
RUN apt-get install -y sudo
RUN apt-get install -y python-opengl
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
RUN mkdir -p /home/developer
ENV HOME /home/developer
WORKDIR /home/developer


#have this in the end because we are not root afterwards
RUN export uid=1000 gid=1001 && \
    mkdir -p /etc/sudoers.d && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${gid}:" >> /etc/group && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer && \
    chown ${uid}:${gid} -R /home/developer
USER developer


