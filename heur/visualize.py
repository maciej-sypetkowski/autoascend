import cv2
import numpy as np


def draw_grid(imgs, ncol):
    grid = imgs.reshape((-1, ncol, *imgs[0].shape))
    rows = []
    for row in grid:
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)
    return img


class DrawPathScope():

    def __init__(self, visualizer, path, color):
        self.visualizer = visualizer
        self.path = path
        self.color = color

    def draw_fun(self, rendered):
        for p1, p2 in zip(self.path, self.path[1:]):
            p1 = [round((i + 0.5) * self.visualizer.tile_size) for i in p1][::-1]
            p2 = [round((i + 0.5) * self.visualizer.tile_size) for i in p2][::-1]
            cv2.line(rendered, p1, p2, self.color, 2)
        return rendered

    def __enter__(self):
        self.fun_instance = lambda x: self.draw_fun(x)
        self.visualizer.drawers.add(self.fun_instance)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.visualizer.drawers.remove(self.fun_instance)


class Visualizer:

    def __init__(self, env, tileset_path='/workspace/heur/tilesets/3.6.1tiles32.png', tile_size=32):
        self.env = env
        self.tile_size = tile_size

        self.tileset = cv2.imread(tileset_path)[..., ::-1]
        if self.tileset is None:
            raise FileNotFoundError(f'Tileset {tileset_path} not found')
        if self.tileset.shape[0] % tile_size != 0 or self.tileset.shape[1] % tile_size != 0:
            raise ValueError("Tileset and tile_size doesn't match modulo")

        h = self.tileset.shape[0] // tile_size
        w = self.tileset.shape[1] // tile_size
        tiles = []
        for y in range(h):
            y *= tile_size
            for x in range(w):
                x *= tile_size
                tiles.append(self.tileset[y:y + tile_size, x:x + tile_size])
        self.tileset = np.array(tiles)

        from glyph2tile import glyph2tile
        self.glyph2tile = np.array(glyph2tile)

        print('Read tileset of size:', self.tileset.shape)

        cv2.namedWindow('NetHackVis')

        self.drawers = set()

    def debug_path(self, path, color):
        return DrawPathScope(self, path, color)

    def update(self, obs):
        glyphs = obs['glyphs']
        tiles_idx = self.glyph2tile[glyphs]
        tiles = self.tileset[tiles_idx.reshape(-1)]
        rendered = draw_grid(tiles, glyphs.shape[1])
        for drawer in self.drawers:
            rendered = drawer(rendered)
        cv2.imshow('NetHackVis', rendered[..., ::-1])
        cv2.waitKey(1)
