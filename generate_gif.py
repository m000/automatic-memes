#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import dlib
import gizeh as gz
import logging
import moviepy.editor as mpy
import numpy as np
import shlex
import sys

from imutils import face_utils
from pathlib import Path
from PIL import Image

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68.dat')

class NoFacesDetectedError(Exception):
    pass

class DealGifFace:
    def __init__(self, dealgif, rect, no_rotate=False):
        self.dealgif = dealgif
        self.rect = rect
        self.no_rotate = no_rotate
        logging.debug("face-rect: %s" % ((self.top_left, self.bottom_right),))

        # extract facial features
        # see: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
        self.features = face_utils.shape_to_np(
                predictor(self.dealgif.bgr_gray, self.rect))

        # initialize swag
        self.set_swag()

    def set_swag(self, swagimg=None):
        """ Rotate swag image and fit it to eyes centers.
        """
        if swagimg is None:
            swagimg = self.dealgif.swag
        if getattr(self, 'swag', None) is not None:
            self.swag.close()

        # scale swag to fit the face
        sw = self.width
        sh = (self.width * swagimg.size[1]) // swagimg.size[0]
        self.swag = swagimg.resize((sw, sh), resample=Image.LANCZOS)
        self.swag=self.swag.transpose(Image.FLIP_LEFT_RIGHT)
        if not self.no_rotate:
            self.swag=self.swag.rotate(self.eyes_angle(), expand=True)

        # shift swag to the leftmost position of the left eye
        left_eye_x = self.left_eye[0,0] - int(self.width / 4.5)
        left_eye_y = self.left_eye[0,1] - int(self.width / 7.5)
        self.swag_pos = (left_eye_x, left_eye_y)

    def eyes_angle(self):
        left_eye_c = self.right_eye.mean(axis=0).astype('int')
        right_eye_c = self.left_eye.mean(axis=0).astype('int')
        dy = left_eye_c[1] - right_eye_c[1]
        dx = left_eye_c[0] - right_eye_c[0]
        return -np.rad2deg(np.arctan2(dy, dx))

    @property
    def left_eye(self):
        return self.features[36:42]
    @property
    def right_eye(self):
        return self.features[42:48]
    @property
    def width(self):
        return self.rect.right() - self.rect.left()
    @property
    def top_left(self):
        return (self.rect.top(), self.rect.left())
    @property
    def bottom_right(self):
        return (self.rect.bottom(), self.rect.right())


class DealGif:
    TEXT_DT = 1
    END_DT = 2.5
    def __init__(self, bgr, duration=4, fps=4, max_width=500,
            suffix='.gif', keep_faces=[], no_rotate=False,
            text_top=None, text_bottom=None, font_size=50):
        self.bgrpath = Path(bgr)
        self.bgr = Image.open(self.bgrpath.as_posix()).convert('RGBA')
        self.duration = duration
        self.fps = fps
        self.suffix = suffix
        self.keep_faces = keep_faces
        self.no_rotate = no_rotate
        self.font_size = font_size

        # scale if needed
        if max_width > 0 and self.bgr.size[0] > max_width:
            thumbw = max_width
            thumbh = (thumbw * self.bgr.size[1]) // self.bgr.size[0]
            self.bgr.thumbnail((thumbw, thumbh), Image.ANTIALIAS)

        # convert to gray for dlib face detector
        self.bgr_gray = np.array(self.bgr.convert('L'))

        # render text
        self.text_top = text_top
        self.text_bottom = text_bottom
        self.gztext = []
        if self.text_top is not None:
            self.gztext.append(self.make_text(self.text_top, 'top'))
        if self.text_bottom is not None:
            self.gztext.append(self.make_text(self.text_bottom, 'bottom'))

        # uninitialized members
        self.faces = []
        self.swag = None
        self.animation = None

    @property
    def outpath(self):
        return self.bgrpath.with_name('%s-deal'
                % (self.bgrpath.stem)).with_suffix(self.suffix)

    def make_text(self, text, where='bottom'):
        if text is None:
            return None
        if where == 'top':
            xy=(self.bgr.size[0] // 2, int(self.bgr.size[1] * 0.1))
        else:
            xy=(self.bgr.size[0] // 2, int(self.bgr.size[1] * 0.9))
        return gz.text(text, xy=xy,
                fontfamily='Impact', fontsize=self.font_size, fontweight='bold',
                fill=(1, 1, 1), stroke=(0, 0, 0), stroke_width=2)

    def make_faces(self):
        # detect bounding rectangles for faces and sort them left to right, top to bottom.
        face_rects = sorted(detector(self.bgr_gray, 0), key=lambda r: (r.left(), r.top()))

        if not face_rects:
            raise NoFacesDetectedError(self.bgrpath)

        # create face objects, keeping only the filtered faces
        if self.keep_faces == []:
            self.faces = [DealGifFace(self, r, self.no_rotate) for r in face_rects]
        else:
            self.faces = [DealGifFace(self, face_rects[i-1], self.no_rotate) for i in self.keep_faces]

    def make_frame(self, t):
        # Make an RGB copy of the background image to work with.
        # make_frame() is expected to return RGB images. RGBA works
        # ok for gif output, but results in garbled video output.
        # https://zulko.github.io/moviepy/ref/VideoClip/VideoClip.html#moviepy.video.VideoClip.VideoClip
        frame = self.bgr.convert('RGB')

        t_swag_start = 0.10 * self.duration
        t_swag_end = self.duration

        if t <= t_swag_start:
            # no swag for starting frames
            pass
        elif t <= t_swag_end:
            # moving swag
            for face in self.faces:
                current_x = face.swag_pos[0]
                current_y = int(face.swag_pos[1] * (t - t_swag_start) / (t_swag_end - t_swag_start))
                frame.paste(face.swag, (current_x, current_y), face.swag)
        else:
            # stable swag
            for face in self.faces:
                frame.paste(face.swag, face.swag_pos, face.swag)

            # draw text
            if len(self.gztext) > 0:
                for gzt in self.gztext[0:self.gztext_i+1]:
                    gzs = gz.Surface(*self.bgr.size, bg_color=None)
                    gzt.draw(gzs)
                    gzs_image = Image.fromarray(gzs.get_npimage(transparent=True))
                    frame.paste(gzs_image, (0, 0), gzs_image)
                    gzs_image.close()

                if t > t_swag_end + (self.gztext_i + 1)*DealGif.TEXT_DT:
                    self.gztext_i += 1

        return np.asarray(frame)

    def make_animation(self):
        # increase duration to show ending text or pause to final frame
        duration = self.duration
        if len(self.gztext) > 0:
            duration += DealGif.TEXT_DT*len(self.gztext)
        duration += DealGif.END_DT
        self.gztext_i = 0
        self.animation = mpy.VideoClip(self.make_frame, duration=duration)

    def write(self, outpath=None):
        outpath = self.outpath.as_posix() if outpath is None else outpath
        self.animation.set_duration(self.duration)
        if self.suffix == '.gif':
            self.animation.write_gif(outpath, colors=256, fps=args.fps)
        else:
            # video_debug = {'verbose': True, 'remove_temp': False, 'write_logfile': True}
            video_debug = {}
            self.animation.write_videofile(outpath, fps=24, preset='slow', **video_debug)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="automatic deal-with-it generator",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--swag-img", default="deals.png",
            help="deal-with-it swag image")
    parser.add_argument("--text-bottom", default="DEAL WITH IT",
            help="set bottom text")
    parser.add_argument("--text-top", default=None,
            help="set optional top text")
    parser.add_argument("--max-width", type=int, default=500,
            help="maximum width for output -- keep gif size small")
    parser.add_argument("--duration", type=int, default=4,
            help="duration for the output file")
    parser.add_argument("--fps", type=int, default=4,
            help="fps for the output file")
    parser.add_argument("--font-pt", type=int, default=50,
            help="font size for text")
    parser.add_argument("--suffix", default='.gif',
            help="set the output file type")
    parser.add_argument('--no-rotate', action='store_true',
            help="Don't rotate the swag image.")
    parser.add_argument('--keep-faces', nargs='+', type=int,
            default=[], metavar='FIDX',
            help="Keep only the faces with the specified indexes.")

    args, uargs = parser.parse_known_args()

    swag_img = Image.open(args.swag_img)
    print(args)
    for ua in uargs:
        try:
            deal_gif = DealGif(ua,
                    max_width=args.max_width, font_size=args.font_pt,
                    keep_faces=args.keep_faces, no_rotate=args.no_rotate,
                    text_top = args.text_top, text_bottom = args.text_bottom,
                    duration=args.duration, fps=args.fps, suffix=args.suffix)
            deal_gif.swag = swag_img
            deal_gif.make_faces()
        except FileNotFoundError:
            logging.warning("skipping %s -- not a file", ua)
            continue
        except OSError:
            logging.warning("skipping %s -- not an image", ua)
            continue
        except NoFacesDetectedError:
            logging.warning("skipping %s -- no faces detected", ua)
            continue

        logging.info("processing %s -- %d face(s) found",
                deal_gif.bgrpath, len(deal_gif.faces))
        deal_gif.make_animation()
        deal_gif.write()

# vim: expandtab:ts=4:sts=4:sw=4:
