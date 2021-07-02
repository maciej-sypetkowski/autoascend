import numpy as np
import numba as nb


@nb.njit
def bfs(glyphs, walkable, walkable_diagonally, y, x):
    SIZE_X = 79 # C.SIZE_X
    SIZE_Y = 21 # C.SIZE_Y

    dis = np.zeros((SIZE_Y, SIZE_X), dtype=np.int16)
    dis[:] = -1
    dis[y, x] = 0

    buf = np.zeros((SIZE_Y * SIZE_X, 2), dtype=np.uint16)
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
        # TODO: debug diagonal moving into and from doors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                py, px = y + dy, x + dx
                if 0 <= py < SIZE_Y and 0 <= px < SIZE_X and (dy != 0 or dx != 0):
                    if (walkable[py, px] and
                            (abs(dy) + abs(dx) <= 1 or (walkable_diagonally[py, px] and walkable_diagonally[y, x]))):
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
    ret[sy : sy + array.shape[0], sx : sx + array.shape[1]] = array
    return ret
