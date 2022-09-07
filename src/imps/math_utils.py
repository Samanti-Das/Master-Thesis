import numpy as np


def rotX(th, tr):
    if tr:
        R = np.array([
            [1,      0    ,     0      , 0],
            [0, np.cos(th), -np.sin(th), 0],
            [0, np.sin(th),  np.cos(th), 0],
            [    0      , 0,     0     , 1]
        ])

    else:
        R = np.array([
            [1,      0    ,     0      ],
            [0, np.cos(th), -np.sin(th)],
            [0, np.sin(th),  np.cos(th)],
        ])

    return R


def rotY(th, tr):
    if tr:
        R = np.array([
            [np.cos(th) , 0, np.sin(th), 0],
            [    0      , 1,     0     , 0],
            [-np.sin(th), 0, np.cos(th), 0],
            [    0      , 0,     0     , 1]
        ])

    else:
        R = np.array([
            [np.cos(th) , 0, np.sin(th)],
            [    0      , 1,     0     ],
            [-np.sin(th), 0, np.cos(th)]
        ])

    return R

def rotZ(ph, tr):
    if tr:
        R = np.array([
            [np.cos(ph), -np.sin(ph), 0, 0],
            [np.sin(ph),  np.cos(ph), 0, 0],
            [    0     ,      0     , 1, 0],
            [    0     ,      0     , 0, 1]
        ])

    else:
        R = np.array([
            [np.cos(ph), -np.sin(ph), 0],
            [np.sin(ph),  np.cos(ph), 0],
            [    0     ,      0     , 1]
        ])

    return R