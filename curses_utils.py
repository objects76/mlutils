


import curses as C
from curses import ascii
import io

class curses_attr:
    def __init__(self, screen, attr):
        self.screen = screen
        self.attr = attr
    def __enter__(self):
        self.screen.attron(self.attr)
        return self.screen
    def __exit__(self, exc_type, exc_value, traceback):
        self.screen.attroff(self.attr)


#
#
#
class Win:
    def __init__(self, win) -> None:
        self.win = win

    def size(self):
        h, w = self.win.getmaxyx()
        return (w,h)

    def rect(self):
        h, w = self.win.getmaxyx()
        y, x = self.win.getbegyx()
        return (x,y,x+w,y+h)

    def print(self, *args, **kwargs):
        buf = io.StringIO()
        print(*args, **kwargs, file=buf)
        message = buf.getvalue()
        buf.close()

        w,h = self.size()
        # curx, cury = self.cursor()
        # if cury+1 == h:
        #     self.win.deleteln()

        # message = f"{curx=}, {cury=}: " + message
        self.win.addnstr(message, min(w, len(message)))
        self.win.refresh()

    def update(self):
        self.win.refresh()

    def clear(self):
        self.win.clear()
        self.win.move(0,0)

    def move(self, x=0, y=0):
        self.win.move(y,x)

    def cursor(self):
        y, x = self.win.getyx()
        return (x,y)

    def scroll(self, dy=1):
        self.win.move(0,0)
        for _ in range(dy):
            self.win.deleteln()
        w,h = self.size()
        self.win.move(0, h-1-dy)

#
#
#
class Screen:
    def __init__(self, stdscr) -> None:
        C.curs_set(False)
        self.stdscr = stdscr
        self.stdscr.nodelay(True)
        self.ROWS, self.COLS = self.stdscr.getmaxyx()
        self.stdscr.border()

        self.stdscr.getch() # win update work after first stdscr.getch()

    def size(self):
        self.ROWS, self.COLS = self.stdscr.getmaxyx()
        return (self.COLS, self.ROWS) # w,h

    def newwin(self, width, height, left, top):
        W,H = self.size()

        right = width+left
        bottom = height+top

        right = min(W,right)
        bottom = min(H,bottom + 1) # +1: for nl of last line.

        lines = bottom-top
        width = right-left
        win = C.newwin(lines, width, top, left)
        return win

    def status(self, message):
        W,H = self.size()
        pad = ' '*(W - len(message) -4)
        self.stdscr.addstr(H-1, 2, message+pad)

    def getch(self):
        return self.stdscr.getch()

