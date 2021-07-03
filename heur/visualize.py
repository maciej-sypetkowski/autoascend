import cv2
import numpy as np


def draw_grid(imgs, ncol):
    grid = imgs.reshape((-1, ncol, *imgs[0].shape))
    rows = []
    for row in grid:
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)
    return img


class Visualizer:

    def __init__(self, agent, tileset_path='/workspace/heur/tilesets/3.6.1tiles32.png', tile_size=32):
        self.agent = agent

        self.tileset = cv2.imread(tileset_path)
        if self.tileset is None:
            raise FileNotFoundError(f'Tileset {tileset_path} not found')
        if self.tileset.shape[0] % tile_size != 0 or self.tileset.shape[1] % tile_size != 0:
            raise ValueError("Tileset and tile_size doesn't match modulo")

        h = self.tileset.shape[0] // tile_size
        w = self.tileset.shape[1] // tile_size
        print(h, w)
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

    def update(self):
        glyphs = self.agent.glyphs.copy()
        tiles_idx = self.glyph2tile[glyphs]
        tiles = self.tileset[tiles_idx.reshape(-1)]
        rendered = draw_grid(tiles, glyphs.shape[1])
        cv2.imshow('NetHackVis', rendered)
        cv2.waitKey(1)

