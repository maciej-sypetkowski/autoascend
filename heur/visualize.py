import queue
import cv2
import numpy as np

# avoid importing agent modules here, because it makes agent reloading less reliable

MSG_HISTORY_COUNT = 10
FONT_SIZE = 32
FAST_FRAME_SKIPPING = 16
RENDERS_HISTORY_SIZE = 64


def _put_text(img, text, pos, scale=FONT_SIZE / 35, thickness=1, color=(255, 255, 0), console=False):
    # TODO: figure out how exactly opencv anchors the text
    pos = (pos[0] + FONT_SIZE // 2, pos[1] + FONT_SIZE // 2 + 8)

    if console:
        # TODO: implement equal characters size font
        # scale *= 2
        # font = cv2.FONT_HERSHEY_PLAIN
        font = cv2.FONT_HERSHEY_SIMPLEX
    else:
        font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.putText(img, text, pos, font,
                       scale, color, thickness, cv2.LINE_AA)


def _draw_frame(img, color=(90, 90, 90), thickness=3):
    return cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, thickness)


def _draw_grid(imgs, ncol):
    grid = imgs.reshape((-1, ncol, *imgs[0].shape))
    rows = []
    for row in grid:
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)
    return img


class DrawTilesScope():

    def __init__(self, visualizer, tiles, color, is_path=False, is_heatmap=False, mode='fill'):
        from glyph import C
        self.visualizer = visualizer
        self.is_heatmap = is_heatmap
        self.color = color
        self.mode = mode
        if self.is_heatmap:
            assert not is_path
            assert self.mode == 'fill'
            assert isinstance(self.color, str)
            self.tiles = tiles
        else:
            if isinstance(tiles, np.ndarray) and tiles.shape == (C.SIZE_Y, C.SIZE_X):
                self.tiles = list(zip(*tiles.nonzero()))
            else:
                self.tiles = tiles
            self.is_path = is_path

    def draw_fun(self, rendered):
        if self.is_heatmap:
            if self.is_heatmap:
                grayscale = np.zeros(rendered.shape, dtype=float)
                mask = np.ones_like(grayscale).astype(bool)
                for y in range(self.tiles.shape[0]):
                    for x in range(self.tiles.shape[1]):
                        y1 = y * self.visualizer.tile_size
                        x1 = x * self.visualizer.tile_size
                        if np.isnan(self.tiles[y, x]):
                            mask[y1:y1 + self.visualizer.tile_size,
                                 x1:x1 + self.visualizer.tile_size] = False
                        else:
                            grayscale[y1:y1 + self.visualizer.tile_size,
                                      x1:x1 + self.visualizer.tile_size] = self.tiles[y, x]
                grayscale[mask] -= np.min(grayscale[mask])
                grayscale[mask] /= np.max(grayscale[mask])
                grayscale = (grayscale * 255).astype(np.uint8)
                grayscale = cv2.blur(grayscale, (15, 15))
                # https://docs.opencv.org/4.5.2/d3/d50/group__imgproc__colormap.html
                heatmap = cv2.applyColorMap(grayscale, cv2.__dict__[f'COLORMAP_{self.color.upper()}'])[..., ::-1]
                return (rendered * 0.5 + heatmap * 0.5).astype(np.uint8) * mask + (rendered * ~mask)

        color = self.color
        alpha = 1
        if len(color) == 4:
            alpha = color[-1] / 255
            color = color[:-1]

        if alpha != 1:
            orig_rendered, rendered = rendered, np.zeros_like(rendered)

        if self.is_path:
            for p1, p2 in zip(self.tiles, self.tiles[1:]):
                p1 = [round((i + 0.5) * self.visualizer.tile_size) for i in p1][::-1]
                p2 = [round((i + 0.5) * self.visualizer.tile_size) for i in p2][::-1]
                cv2.line(rendered, p1, p2, color, 2)
        else:
            for p in self.tiles:
                p1 = [round(i * self.visualizer.tile_size) for i in p][::-1]
                p2 = [round((i + 1) * self.visualizer.tile_size) for i in p][::-1]
                if self.mode == 'fill':
                    cv2.rectangle(rendered, p1, p2, color, -1)
                if self.mode == 'frame':
                    cv2.rectangle(rendered, p1, p2, color, 3)

        if alpha != 1:
            rendered = np.clip(orig_rendered.astype(np.int16) + (rendered * alpha).astype(np.int16), 0, 255).astype(
                np.uint8)

        return rendered.copy()

    def __enter__(self):
        self.fun_instance = lambda x: self.draw_fun(x)
        self.visualizer.drawers.append(self.fun_instance)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.visualizer.drawers.remove(self.fun_instance)


class DebugLogScope():

    def __init__(self, visualizer, txt, color):
        self.visualizer = visualizer
        self.txt = txt
        self.color = color

    def __enter__(self):
        self.visualizer.log_messages.append(self.txt)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.visualizer.log_messages.remove(self.txt)


class Visualizer:

    def __init__(self, env, tileset_path='/workspace/heur/tilesets/3.6.1tiles32.png', tile_size=32,
                 start_visualize=None, show=False, output_dir=None):
        self.env = env
        self.tile_size = tile_size
        self._window_name = 'NetHackVis'
        self.show = show
        self.start_visualize = start_visualize
        self.output_dir = output_dir

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

        if self.show:
            print('Read tileset of size:', self.tileset.shape)

        self.create_window()

        self.message_history = list()
        self.popup_history = list()

        self.drawers = []
        self.log_messages = list()
        self.log_messages_history = list()

        self.frame_skipping = 1
        self.frame_counter = -1

        self.renders_history = None
        if not self.show:
            assert output_dir is not None
            self.renders_history = queue.deque(maxlen=RENDERS_HISTORY_SIZE)
            self.output_dir = output_dir
            self.output_dir.mkdir(exist_ok=True, parents=True)

    def create_window(self):
        if self.show:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)

    def close_window(self):
        if self.show:
            cv2.destroyWindow(self._window_name)

    def debug_tiles(self, *args, **kwargs):
        return DrawTilesScope(self, *args, **kwargs)

    def debug_log(self, txt, color):
        return DebugLogScope(self, txt, color)

    def step(self, obs):
        self.last_obs = obs
        self._update_log_message_history()
        self._update_message_history()
        self._update_popup_history()

    def render(self):
        self.frame_counter += 1

        if self.frame_counter % self.frame_skipping != 0:
            return

        if self.last_obs is None:
            return

        if self.start_visualize is not None:
            if self.env.step_count < self.start_visualize:
                return

        glyphs = self.last_obs['glyphs']
        tiles_idx = self.glyph2tile[glyphs]
        tiles = self.tileset[tiles_idx.reshape(-1)]
        scene_vis = _draw_grid(tiles, glyphs.shape[1])
        for drawer in self.drawers:
            scene_vis = drawer(scene_vis)
        _draw_frame(scene_vis)
        topbar = self._draw_topbar(self.last_obs, scene_vis.shape[1])
        tty = self._draw_tty(self.last_obs, scene_vis.shape[1])

        rendered = np.concatenate([topbar, scene_vis, tty], axis=0)
        inventory = self._draw_inventory(rendered.shape[0])
        rendered = np.concatenate([rendered, inventory], axis=1)

        if self.show:
            cv2.imshow(self._window_name, rendered[..., ::-1])
            cv2.waitKey(1)

        if self.renders_history is not None:
            self.renders_history.append(rendered)

    def _draw_topbar(self, obs, width):
        messages_vis = self._draw_message_history(width // 3)
        popup_vis = self._draw_popup_history(width // 4)
        log_messages_vis = self._draw_debug_message_log(width - width // 3 - width // 4)
        ret = np.concatenate([messages_vis, popup_vis, log_messages_vis], axis=1)
        assert ret.shape[1] == width
        return ret

    def _draw_debug_message_log(self, width):
        vis = np.zeros((FONT_SIZE * MSG_HISTORY_COUNT, width, 3)).astype(np.uint8)
        for i in range(MSG_HISTORY_COUNT):
            if i >= len(self.log_messages_history):
                break
            txt = self.log_messages_history[-i - 1]
            if i == 0:
                _put_text(vis, txt, (0, i * FONT_SIZE), color=(255, 255, 255))
            else:
                _put_text(vis, txt, (0, i * FONT_SIZE), color=(120, 120, 120))
        _draw_frame(vis)
        return vis

    def _update_log_message_history(self):
        txt = ''
        if self.env.agent is not None:
            txt = ' | '.join(self.log_messages)
        # if txt:
        self.log_messages_history.append(txt)

    def _draw_message_history(self, width):
        messages_vis = np.zeros((FONT_SIZE * MSG_HISTORY_COUNT, width, 3)).astype(np.uint8)
        for i in range(MSG_HISTORY_COUNT):
            if i >= len(self.message_history):
                break
            txt = self.message_history[-i - 1]
            if i == 0:
                _put_text(messages_vis, txt, (0, i * FONT_SIZE), color=(255, 255, 255))
            else:
                _put_text(messages_vis, txt, (0, i * FONT_SIZE), color=(120, 120, 120))
        _draw_frame(messages_vis)
        return messages_vis

    def _draw_popup_history(self, width):
        messages_vis = np.zeros((FONT_SIZE * MSG_HISTORY_COUNT, width, 3)).astype(np.uint8)
        for i in range(MSG_HISTORY_COUNT):
            if i >= len(self.popup_history):
                break
            txt = '|'.join(self.popup_history[-i - 1])
            if i == 0:
                _put_text(messages_vis, txt, (0, i * FONT_SIZE), color=(255, 255, 255))
            else:
                _put_text(messages_vis, txt, (0, i * FONT_SIZE), color=(120, 120, 120))
        _draw_frame(messages_vis)
        return messages_vis

    def _update_message_history(self):
        txt = ''
        if self.env.agent is not None:
            txt = self.env.agent.message
        # if txt:
        self.message_history.append(txt)

    def _update_popup_history(self):
        txt = ''
        if self.env.agent is not None:
            txt = self.env.agent.popup
        # if txt:
        self.popup_history.append(txt)

    def _draw_tty(self, obs, width):
        vis = np.zeros((FONT_SIZE * len(obs['tty_chars']), width, 3)).astype(np.uint8)
        for i, line in enumerate(obs['tty_chars']):
            txt = ''.join([chr(i) for i in line])
            _put_text(vis, txt, (0, i * FONT_SIZE), console=True)
        _draw_frame(vis)
        return vis

    def _draw_item(self, letter, item, width, height):
        from item import Item

        vis = np.zeros((height, width, 3)).astype(np.uint8)
        _draw_frame(vis, color=(50, 50, 50), thickness=2)
        _put_text(vis, str(letter), (0, 0))
        status_str, status_col = {
            Item.UNKNOWN: (' ', (255, 255, 255)),
            Item.CURSED: ('C', (255, 0, 0)),
            Item.UNCURSED: ('U', (0, 255, 255)),
            Item.BLESSED: ('B', (0, 255, 0)),
        }[item.status]
        _put_text(vis, str(letter), (0, 0))
        _put_text(vis, status_str, (FONT_SIZE, 0), color=status_col)
        if item.modifier is not None:
            _put_text(vis, str(item.modifier), (FONT_SIZE * 2, 0))

        if item.is_weapon():
            _put_text(vis, str(self.env.agent.character.get_melee_bonus(item)), (FONT_SIZE * 4, 0))

        _put_text(vis, str(item), (FONT_SIZE * 8, 0))
        if item.equipped:
            cv2.rectangle(vis, (0, 0), (int(FONT_SIZE * 1.5), vis.shape[0] - 1), (0, 255, 255), 6)
        return vis

    def _draw_inventory(self, height):
        width = 800
        vis = np.zeros((height, width, 3)).astype(np.uint8)
        if self.env.agent:
            item_h = FONT_SIZE
            for i, (letter, item) in enumerate(zip(self.env.agent.inventory.items.all_letters,
                                                   self.env.agent.inventory.items.all_items)):
                vis[i * item_h:(i + 1) * item_h] = self._draw_item(letter, item, width, item_h)
        _draw_frame(vis)
        return vis

    def save_end_history(self):
        for i, render in enumerate(list(self.renders_history)):
            render = render[..., ::-1]
            out_path = self.output_dir / (str(i).rjust(5, '0') + '.jpg')
            cv2.imwrite(str(out_path), render)