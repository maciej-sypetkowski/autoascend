import cv2
import numpy as np

MSG_HISTORY_COUNT = 6
FONT_SIZE = 32
FAST_FRAME_SKIPPING = 8


def put_text(img, text, pos, scale=FONT_SIZE / 55, thickness=1, color=(255, 255, 0), console=False):
    # TODO: figure out how exactly opencv anchors the text
    pos = (pos[0] + FONT_SIZE // 2 + 2, pos[1] + FONT_SIZE // 2 + 2)

    if console:
        # TODO: implement equal characters size font
        # scale *= 2
        # font = cv2.FONT_HERSHEY_PLAIN
        font = cv2.FONT_HERSHEY_SIMPLEX
    else:
        font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.putText(img, text, pos, font,
                       scale, color, thickness, cv2.LINE_AA)


def draw_frame(img, color=(90, 90, 90), thickness=3):
    return cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, thickness)


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

        cv2.namedWindow('NetHackVis', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)

        self.message_history = list()

        self.drawers = set()

        self.frame_skipping = 1
        self.frame_counter = -1

    def debug_path(self, path, color):
        return DrawPathScope(self, path, color)

    def draw_topbar(self, obs, width):
        messages_vis = np.zeros((FONT_SIZE * MSG_HISTORY_COUNT, width // 2, 3)).astype(np.uint8)
        txt = ''
        if self.env.agent is not None:
            txt = self.env.agent.message
        if txt:
            self.message_history.append(txt)
        for i in range(MSG_HISTORY_COUNT):
            if i >= len(self.message_history):
                break
            txt = self.message_history[-i - 1]
            if i == 0:
                put_text(messages_vis, txt, (0, i * FONT_SIZE), color=(255, 255, 255))
            else:
                put_text(messages_vis, txt, (0, i * FONT_SIZE), color=(120, 120, 120))
        draw_frame(messages_vis)

        blstats = np.zeros((FONT_SIZE * MSG_HISTORY_COUNT, width - width // 2, 3)).astype(np.uint8)
        txt = ''
        put_text(blstats, txt, (0, 0), color=(255, 255, 255))
        draw_frame(blstats)

        return np.concatenate([messages_vis, blstats], axis=1)

    def draw_tty(self, obs, width):
        vis = np.zeros((FONT_SIZE * len(obs['tty_chars']), width, 3)).astype(np.uint8)
        for i, line in enumerate(obs['tty_chars']):
            txt = ''.join([chr(i) for i in line])
            put_text(vis, txt, (0, i * FONT_SIZE), console=True)
        draw_frame(vis)
        return vis

    def update(self, obs):
        self.frame_counter += 1
        if self.frame_counter % self.frame_skipping != 0:
            return

        glyphs = obs['glyphs']
        tiles_idx = self.glyph2tile[glyphs]
        tiles = self.tileset[tiles_idx.reshape(-1)]
        scene_vis = draw_grid(tiles, glyphs.shape[1])
        for drawer in self.drawers:
            scene_vis = drawer(scene_vis)
        draw_frame(scene_vis)
        topbar = self.draw_topbar(obs, scene_vis.shape[1])
        tty = self.draw_tty(obs, scene_vis.shape[1])
        rendered = np.concatenate([topbar, scene_vis, tty], axis=0)
        self.frame_counter += 1

        cv2.imshow('NetHackVis', rendered[..., ::-1])
        cv2.waitKey(1)
