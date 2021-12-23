import av
import cv2
import numpy as np

import logging
from argparse import ArgumentParser
from math import ceil
from multiprocessing import Process, Queue
from time import time


def parseArguments():
    r'''Parse command-line arguments.
    '''
    parser = ArgumentParser(description='Video player that can reproduce multiple files simultaneoulsy')
    parser.add_argument('ip', help='IP:port or FQDN:port of the rtsp stream')
    parser.add_argument('username', default='', help='username for the rtsp stream')
    parser.add_argument('password', default='', help='password for the rtsp stream')
    parser.add_argument('--channels', nargs='+', default=['1','3','4','6','8','9'], help='Channels to tune into on this stream')
    parser.add_argument('--resolution', type=int, nargs=2, default=[1920, 1080], help='Resolution of the combined video')
    parser.add_argument('--fps', type=int, default=15, help='Frame rate used when playing video contents')

    return parser.parse_args()


def decode(path, width, height, queue):
    r'''Decode a video and return its frames through a process queue.

        Frames are resized to `(width, height)` before returning.
    '''
    container = av.open(path)
    for frame in container.decode(video=0):
        # TODO: Keep image ratio when resizing.
        image = frame.to_rgb(width=width, height=height).to_ndarray()
        queue.put(image)

    queue.put(None)


class GridViewer(object):
    r'''Interface for displaying video frames in a grid pattern.
    '''
    def __init__(self, args):
        r'''Create a new grid viewer.
        '''
        size = float(6.0)
        self.cols = ceil(size ** 0.5)
        self.rows = ceil(size / self.cols)

        (width, height) = args.resolution
        self.shape = (height, width, 3)

        self.cell_width = width // self.cols
        self.cell_height = height // self.rows

        cv2.namedWindow('Video', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow('Video', width, height)

    def update(self, queues):
        r'''Query the frame queues and update the viewer.

            Return whether all decoders are still active.
        '''
        grid = np.zeros(self.shape, dtype=np.uint8)
        for (k, queue) in enumerate(queues):
            image = queue.get()
            if image is None:
                return False

            (i, j) = (k // self.cols, k % self.cols)
            (m, n) = image.shape[:2]

            a = i * self.cell_height
            b = a + m

            c = j * self.cell_width
            d = c + n

            grid[a:b, c:d] = image

        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', grid)
        cv2.waitKey(1)

        return True


def play(args):
    r'''Play multiple video files in a grid interface.
    '''
    grid = GridViewer(args)

    queues = []
    processes = []
    for channel in args.channels:
        url = 'rtsp://'+ args.username + ':' + args.password + '@' + args.ip + '//cam/realmonitor?channel='+ channel + '&subtype=1'

        queues.append(Queue(1))
        processes.append(Process(target=decode, args=(url, grid.cell_width, grid.cell_height, queues[-1]), daemon=True))
        processes[-1].start()

    period = 1.0 / args.fps
    t_start = time()
    t_frame = 0

    while grid.update(queues):
        # Spin-lock the thread as necessary to maintain the frame rate.
        while t_frame > time() - t_start:
            pass

        t_frame += period

    # Terminate any lingering processes, just in case.
    for process in processes:
        process.terminate()


def main():
    logging.disable(logging.WARNING)

    play(parseArguments())


if __name__ == '__main__':
    main()
