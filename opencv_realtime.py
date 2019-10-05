#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import cv2
import dlib
import gizeh as gz
import imutils
import logging
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from imutils import face_utils, translate, rotate
from imutils.video import VideoStream

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description="live deal-with-it generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--src", type=int, default=0,
        help="select video source to use")
parser.add_argument("--max-width", type=int, default=500,
        help="maximum frame width -- faster processing")
parser.add_argument('--fullscreen', action='store_true',
        help="use fullscreen mode")

args, uargs = parser.parse_known_args()

logging.info("Using video from source #%d.", args.src)

vs = VideoStream(src=args.src)
fps = vs.stream.stream.get(cv2.CAP_PROP_FPS) # need this for animating proper duration

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68.dat')


animation_length = fps * 5
current_animation = 0
glasses_on = fps * 3

deal = Image.open("deals.png")
text = None

dealing = False

if args.fullscreen:
    logging.info("Creating full screen window.")
    cv2.namedWindow('deal generator', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('deal generator', cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN)
else:
    cv2.namedWindow('deal generator', cv2.WINDOW_NORMAL)

frameno = -1
vs.start()
while True:
    frameno += 1
    frame = vs.read()
    frame = imutils.resize(frame, width=args.max_width)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(img_gray, 0)
    faces = []

    # text doesn't change, just initialize it once
    if text is None:
        gzs = gz.Surface(*img.size, bg_color=None)
        gztext = gz.text('DEAL WITH IT', fontfamily="Impact",
            fontsize=50, fontweight='bold',
            xy=(img.size[0] // 2, int(img.size[1] * 0.9)),
            fill=(1, 1, 1), stroke=(0, 0, 0), stroke_width=2)
        gztext.draw(gzs)
        text = Image.fromarray(gzs.get_npimage(transparent=True))

    logging.info("%d faces found in frame #%d", len(rects), frameno)
    for rect in rects:
        face = {}
        shades_width = rect.right() - rect.left()

        # predictor used to detect orientation in place where current face is
        shape = predictor(img_gray, rect)
        shape = face_utils.shape_to_np(shape)

        # grab the outlines of each eye from the input image
        leftEye = shape[36:42]
        rightEye = shape[42:48]

        # compute the center of mass for each eye
        leftEyeCenter = leftEye.mean(axis=0).astype("int")
        rightEyeCenter = rightEye.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = leftEyeCenter[1] - rightEyeCenter[1] 
        dX = leftEyeCenter[0] - rightEyeCenter[0]
        angle = np.rad2deg(np.arctan2(dY, dX)) 

        current_deal = deal.resize((shades_width, int(shades_width * deal.size[1] / deal.size[0])),
                               resample=Image.LANCZOS)
        current_deal = current_deal.rotate(angle, expand=True)
        current_deal = current_deal.transpose(Image.FLIP_TOP_BOTTOM)

        face['glasses_image'] = current_deal
        left_eye_x = leftEye[0,0] - shades_width // 4
        left_eye_y = leftEye[0,1] - shades_width // 6
        face['final_pos'] = (left_eye_x, left_eye_y)

        # I got lazy, didn't want to bother with transparent pngs in opencv
        # this is probably slower than it should be
        if dealing:
            if current_animation < glasses_on:
                current_y = int(current_animation / glasses_on * left_eye_y)
                img.paste(current_deal, (left_eye_x, current_y), current_deal)
            else:
                img.paste(current_deal, (left_eye_x, left_eye_y), current_deal)
                img.paste(text, (0, 0), text)

    if dealing:
        current_animation += 1
        # uncomment below to save pngs for creating gifs, videos
        #img.save("images/%05d.png" % current_animation)
        if current_animation > animation_length:
            dealing = False
            current_animation = 0
        else:
            frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    cv2.imshow("deal generator", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    if key == ord("d"):
        dealing = not dealing

cv2.destroyAllWindows()
vs.stop()

# vim: expandtab:ts=4:sts=4:sw=4:
