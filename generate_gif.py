#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import dlib
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
    def __init__(self, dealgif, rect):
        self.dealgif = dealgif
        self.rect = rect
        logging.debug("face-rect: %s" % ((self.top_left, self.bottom_right),))

        # extract facial features
        # see: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
        self.features = face_utils.shape_to_np(
                predictor(self.dealgif.img_gray, self.rect))

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
        self.swag.rotate(self.eyes_angle(), expand=True)
        self.swag.transpose(Image.FLIP_TOP_BOTTOM)

        # shift swag the leftmost position of the left eye
        left_eye_x = self.left_eye[0,0] - self.width // 4
        left_eye_y = self.left_eye[0,1] - self.width // 6
        self.swag_pos = (left_eye_x, left_eye_y)

    def eyes_angle(self):
        left_eye_c = self.right_eye.mean(axis=0).astype('int')
        right_eye_c = self.left_eye.mean(axis=0).astype('int')
        dy = left_eye_c[1] - right_eye_c[1]
        dx = left_eye_c[0] - right_eye_c[0]
        return np.rad2deg(np.arctan2(dy, dx))

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
    def __init__(self, im, duration, fps, max_width=500):
        self.imgpath = Path(im)
        self.gifpath = self.imgpath.with_name('%s-deal'
                % (self.imgpath.stem)).with_suffix('.gif')
        self.img = Image.open(self.imgpath.as_posix()).convert('RGBA')
        self.duration = duration
        self.fps = fps

        # scale if needed
        if max_width > 0 and self.img.size[0] > max_width:
            thumbw = max_width
            thumbh = (thumbw * self.img.size[1]) // self.img.size[0]
            self.img.thumbnail((thumbw, thumbh), Image.ANTIALIAS)

        # convert to gray for dlib face detector
        self.img_gray = np.array(self.img.convert('L'))

        # initialize faces array
        self.faces = []

        self.swag = None
        self.text = None

    def make_faces(self):
        self.faces = [DealGifFace(self, r) for r in detector(self.img_gray, 0)]
        if not self.faces:
            raise NoFacesDetectedError(self.imgpath)

    def make_frame(self, t):
        img = self.img.convert('RGBA')

        t_swag_start = 0.10 * self.duration
        t_swag_end = 0.75 * self.duration

        if t <= t_swag_start:
            # no swag for starting frames
            pass
        elif t <= t_swag_end:
            # moving swag
            for face in self.faces:
                current_x = face.swag_pos[0]
                #current_y = int(face.swag_pos[1] * n / (self.duration - 2))
                current_y = int(face.swag_pos[1] * (t - t_swag_start) / (t_swag_end - t_swag_start))
                #logging.info((t, current_x, current_y))
                img.paste(face.swag, (current_x, current_y), face.swag)
        else:
            # stable swag + text
            for face in self.faces:
                img.paste(face.swag, face.swag_pos, face.swag)
                text_x = (img.width - self.text.width) // 2
                text_y = img.height - int(self.text.height * 1.1)
                img.paste(self.text, (text_x, text_y), self.text)

        return np.asarray(img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="automatic deal-with-it generator",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--swag-img", default="deals.png",
            help="deal-with-it swag image")
    parser.add_argument("--text-img", default="text.png",
            help="deal-with-it text image")
    parser.add_argument("--max-width", type=int, default=500,
            help="maximum width for output -- keep gif size small")
    parser.add_argument("--duration", type=int, default=4,
            help="duration for the output file")
    parser.add_argument("--fps", type=int, default=4,
            help="fps for the output file")
    args, uargs = parser.parse_known_args()

    swag_img = Image.open(args.swag_img)
    text_img = Image.open(args.text_img)

    for ua in uargs:
        try:
            deal_gif = DealGif(ua, args.duration, args.fps, max_width=args.max_width)
            deal_gif.swag = swag_img
            deal_gif.text = text_img
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
                deal_gif.imgpath, len(deal_gif.faces))

        animation = mpy.VideoClip(deal_gif.make_frame, duration=args.duration)
        animation.write_gif(deal_gif.gifpath, fps=args.fps)

# vim: expandtab:ts=4:sts=4:sw=4:
