from functools import partial, wraps
from itertools import chain

import numba as nb
import numpy as np
import toolz

from strategy import Strategy


@nb.njit(cache=True)
def bfs(y, x, *, walkable, walkable_diagonally):
    dis = np.zeros(walkable.shape, dtype=np.int32)
    dis[:] = -1
    dis[y, x] = 0

    buf = np.zeros((walkable.shape[0] * walkable.shape[1], 2), dtype=np.uint32)
    index = 0
    buf[index] = (y, x)
    size = 1
    while index < size:
        y, x = buf[index]
        index += 1

        # TODO: handle situations
        # dir: SE
        # @|
        # -.
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                py, px = y + dy, x + dx
                if 0 <= py < walkable.shape[0] and 0 <= px < walkable.shape[1] and (dy != 0 or dx != 0):
                    if (walkable[py, px] and
                            (abs(dy) + abs(dx) <= 1 or (walkable_diagonally[py, px] and walkable_diagonally[y, x] and
                                                        (walkable[py, x] or walkable[y, px])))):
                        if dis[py, px] == -1:
                            dis[py, px] = dis[y, x] + 1
                            buf[size] = (py, px)
                            size += 1

    return dis


def translate(array, y_offset, x_offset):
    ret = np.zeros_like(array)

    if y_offset > 0:
        array = array[:-y_offset]
    elif y_offset < 0:
        array = array[-y_offset:]
    if x_offset > 0:
        array = array[:, :-x_offset]
    elif x_offset < 0:
        array = array[:, -x_offset:]

    sy, sx = max(y_offset, 0), max(x_offset, 0)
    ret[sy: sy + array.shape[0], sx: sx + array.shape[1]] = array
    return ret


def isin(array, *elems):
    elems = list(chain(*elems))
    return np.isin(array, elems)


@toolz.curry
def debug_log(txt, fun, color=(255, 255, 255)):
    @wraps(fun)
    def wrapper(self, *args, **kwargs):
        # TODO: make it cleaner
        if type(self).__name__ != 'Agent':
            env = self.agent.env
        else:
            env = self.env

        with env.debug_log(txt=txt, color=color):
            ret = fun(self, *args, **kwargs)
            if isinstance(ret, Strategy):
                def f(strategy=ret.strategy, *a, **k):
                    it = strategy(*a, **k)
                    yield next(it)
                    with env.debug_log(txt=txt, color=color):
                        try:
                            next(it)
                            assert 0
                        except StopIteration as e:
                            return e.value

                ret.strategy = partial(f, ret.strategy)
            return ret

    return wrapper


def adjacent(p1, p2):
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])) == 1


def calc_dps(to_hit, damage):
    return damage * np.clip((to_hit - 1), 0, 20) / 20
