"""Reproduce the before/after image-based error of the intrinsic camera calibration.

Post-hoc check for thesis RQ2 / SO2. It refits the 30 saved checkerboard views
with three camera models and prints the RMS reprojection error of each:
  pinhole, no distortion terms   = image used without lens correction ("before")
  fisheye, as calibrate_app.py   = the thesis calibration ("after", 0.356 px)
  fisheye with D fixed to zero   = how much the four distortion terms add

Board: 9 x 6 inner corners, 25 mm squares (calibration/intrinsics/calibrate_app.py).

Usage (from the repo root):
    python tools/calib_before_after.py
"""

import glob
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, 'calibration', 'intrinsics', 'results', '20260701_180005_calibration')
COLS, ROWS, SQUARE_MM = 9, 6, 25.0

objp = np.zeros((COLS * ROWS, 1, 3), np.float64)
objp[:, 0, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQUARE_MM

obj_pts, img_pts, size = [], [], None
for f in sorted(glob.glob(os.path.join(RUN, 'raw_images', '*.png'))):
    gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    size = gray.shape[::-1]
    found, corners = cv2.findChessboardCornersSB(gray, (COLS, ROWS))
    if found:
        obj_pts.append(objp.copy())
        img_pts.append(corners.reshape(-1, 1, 2).astype(np.float64))
print(f'views {len(obj_pts)}, image size {size}')

criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-9)
base = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
no_dist = (base + cv2.fisheye.CALIB_FIX_K1 + cv2.fisheye.CALIB_FIX_K2
           + cv2.fisheye.CALIB_FIX_K3 + cv2.fisheye.CALIB_FIX_K4)
for name, flags in [('fisheye, as thesis', base), ('fisheye, D fixed to 0', no_dist)]:
    rms, K, D, _, _ = cv2.fisheye.calibrate(obj_pts, img_pts, size, np.zeros((3, 3)), np.zeros((4, 1)),
                                            None, None, flags, criteria)
    print(f'{name:24s} rms {rms:.3f} px   D {np.round(D.ravel(), 4)}')

rms, _, _, _, _ = cv2.calibrateCamera(
    [o.reshape(-1, 3).astype(np.float32) for o in obj_pts],
    [i.reshape(-1, 2).astype(np.float32) for i in img_pts], size, None, None,
    flags=cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3)
print(f'{"pinhole, no distortion":24s} rms {rms:.3f} px')
