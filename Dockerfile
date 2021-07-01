FROM nvcr.io/nvidia/pytorch:21.06-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y build-essential autoconf libtool pkg-config python3-dev python3-pip python3-numpy git flex bison libbz2-dev
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add
RUN apt-get install -y software-properties-common xterm
RUN pip install numpy tensorboard gym ray seaborn nevergrad ray[default]

RUN git clone https://github.com/facebookresearch/nle.git /nle --recursive \
        && cd /nle && git checkout 50163c0bf9a318167cc4d117676ef391d2281097 \
        && sed '/#define NLE_ALLOW_SEEDING 1/i#define NLE_ALLOW_SEEDING 1' /nle/include/nleobs.h -i \
        && sed '/self\.env\.set_initial_seeds = f/d' /nle/nle/env/tasks.py -i \
        && sed '/self\.env\.set_current_seeds = f/d' /nle/nle/env/tasks.py -i \
        && sed '/self\.env\.get_current_seeds = f/d' /nle/nle/env/tasks.py -i \
        && sed '/def seed(self, core=None, disp=None, reseed=True):/d' /nle/nle/env/tasks.py -i \
        && sed '/raise RuntimeError("NetHackChallenge doesn.t allow seed changes")/d' /nle/nle/env/tasks.py -i
RUN cd /nle && python setup.py install
