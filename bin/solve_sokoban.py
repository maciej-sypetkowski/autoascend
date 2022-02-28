import os
import sys
import termios
import tty

from autoascend.soko_solver import convert_map, maps


def main():
    termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        # check answers / manually solve
        for text_map, answer in maps.items():
            sokomap = convert_map(text_map)
            if answer is None:
                answer = []

            answer = answer.copy()
            path = []
            while (sokomap.sokomap == BOULDER).sum() > 0 or len(answer) != 0:
                print(path)
                sokomap.print()

                if len(answer) == 0:
                    y, x = sokomap.pos
                    while 1:
                        sokomap.print((y, x))
                        dir = os.read(sys.stdin.fileno(), 3)
                        mapping = {b'j': (1, 0), b'k': (-1, 0), b'h': (0, -1), b'l': (0, 1)}
                        if dir not in mapping:
                            continue
                        dy, dx = mapping[dir]
                        if sokomap.sokomap[y + dy, x + dx] == EMPTY:
                            y, x = y + dy, x + dx
                        elif sokomap.sokomap[y + dy, x + dx] == BOULDER:
                            y, x = y + dy, x + dx
                            break
                else:
                    (y, x), (dy, dx) = answer[0]
                    answer = answer[1:]

                if sokomap.sokomap[y, x] != BOULDER:
                    print('that is not a boulder!')
                    continue
                if sokomap.sokomap[y - dy, x - dx] != EMPTY:
                    print('you cannot stand to push in this direction!')
                    continue
                if sokomap.bfs()[y - dy, x - dx] == -1:
                    print('you cannot get there!')
                    continue
                if sokomap.sokomap[y + dy, x + dx] not in [EMPTY, TARGET]:
                    print('you cannot push in this direction!')
                    continue

                path.append(((y, x), (dy, dx)))

                sokomap.move(y, x, dy, dx)

                print(path)
                sokomap.print()
        print('OK')
    finally:
        os.system('stty sane')


if __name__ == '__main__':
    main()
