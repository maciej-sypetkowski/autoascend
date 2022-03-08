FROM nvcr.io/nvidia/pytorch:21.08-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y build-essential autoconf libtool pkg-config python3-dev python3-pip python3-numpy git flex \
                       bison libbz2-dev xterm gfortran xdot

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY muzero/muzero.patch .
RUN git clone https://github.com/werner-duvaud/muzero-general /muzero && cd /muzero && git checkout 23a1f691
RUN patch -d /muzero muzero.patch

RUN git clone https://github.com/facebookresearch/nle.git /nle --recursive \
 && cd /nle && git checkout v0.7.3 \
 && sed '/#define NLE_ALLOW_SEEDING 1/i#define NLE_ALLOW_SEEDING 1' /nle/include/nleobs.h -i \
 && sed '/self\.env\.set_initial_seeds = f/d' /nle/nle/env/tasks.py -i \
 && sed '/self\.env\.set_current_seeds = f/d' /nle/nle/env/tasks.py -i \
 && sed '/self\.env\.get_current_seeds = f/d' /nle/nle/env/tasks.py -i \
 && sed '/def seed(self, core=None, disp=None, reseed=True):/d' /nle/nle/env/tasks.py -i \
 && sed '/raise RuntimeError("NetHackChallenge doesn.t allow seed changes")/d' /nle/nle/env/tasks.py -i

RUN cd /nle && python setup.py install

# uncomment for PyPy support
# RUN conda create -n pypy pypy
# RUN printf '#!/bin/bash\nexec conda run --no-capture-output -n pypy pypy "$@"' >/usr/bin/pypy3 \
#  && chmod +x /usr/bin/pypy3 \
#  && ln -s /usr/bin/pypy{3,}
# RUN pypy -m ensurepip && pypy -m pip install --upgrade pip
# RUN pypy -m pip install numpy scipy scikit-build toolz pyinstrument
# RUN git clone https://github.com/opencv/opencv-python /opencv-python
# RUN pypy -m pip install scikit-build
# RUN CMAKE_ARGS="-D PYTHON3_LIBRARY=/opt/conda/envs/pypy/lib/libpypy3-c.so" pypy -m pip install opencv-python
# RUN cd /nle && pypy setup.py install

RUN cd /nle/sys/unix && ./setup.sh
RUN cd /nle/util && sed -i '1s/^/CC := $(CC) -I\/nle\/include -DNOMAIL /' Makefile && make
RUN cd /nle/src && sed -i '1s/^/CC := $(CC) -I\/nle\/include -DNOMAIL /' Makefile && make tile.c
RUN python -c 'from pathlib import Path ; text = Path("/nle/src/tile.c").read_text() ; \
               print("glyph2tile = [", text[text.find("{") + 1 : text.find("};")], "]")' >/glyph2tile.py
RUN cd / && python -c 'import nle ; from glyph2tile import glyph2tile ; \
                       assert isinstance(glyph2tile, list) and len(glyph2tile) == nle.nethack.MAX_GLYPH'

# download tileset
RUN mkdir /tilesets && wget 'https://nethackwiki.com/mediawiki/images/7/73/3.6.1tiles32.png' -P /tilesets

# uncomment to install jupyter vim plugin
# RUN apt update && apt install -y npm
# RUN pip install -U jupyterlab==1.2.14
# RUN jupyter labextension uninstall jupyterlab-jupytext jupyterlab_tensorboard
# RUN jupyter labextension install jupyterlab_vim
