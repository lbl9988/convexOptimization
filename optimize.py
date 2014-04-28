#!/usr/bin/env python2

import sys
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
import itertools
import pylab
import glob
from pylab import plot, show
from numpy import vstack, array, arange
from scipy.cluster.vq import kmeans, vq
from collections import Counter
from cvxopt import matrix, solvers
from cvxopt import spmatrix, sparse


def get_point_dist(p0, p1):

    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def get_dist(coords1, coords2):

    n = len(coords1)
    diff = coords1 - coords2
    dist = 0.0
    for i in range(n):
        dist = dist + math.sqrt(diff[i][0]*diff[i][0] + diff[i][1]*diff[i][1])
    dist = dist/n
    return dist


def compute_moments(dists):

    n = int(len(dists))
    m1 = 0.0
    m2 = 0.0
    m3 = 0.0
    m4 = 0.0
    m5 = 0.0
    m6 = 0.0
    mean = 0.0

    for i in range(n):
        mean = mean + dists[i]
    mean = mean/n

    centered_dists = dists - mean
    for i in range(n):
        m1 = m1 + centered_dists[i]
    m1 = m1/n

    for i in range(n):
        m2 = m2 + (centered_dists[i]-m1)**2
        m3 = m3 + (centered_dists[i]-m1)**3
        m4 = m4 + (centered_dists[i]-m1)**4
    m2 = m2/n
    m3 = m3/n
    m4 = m4/n
    m3 = m3/(math.sqrt(m2))**3
    m4 = m4/(m2**2)
    return [mean, m2, m3, m4]


def read_coords(filename):

    #print "reading from file " + str(filename)
    coords = []
    coordfile = open(filename)
    for line in coordfile:
        words = line.split()
        oneCoord = []
        for i in range(len(words)):
            oneCoord.append(float(words[i]))
        coords.append(oneCoord)
    coordfile.close()
    return coords


def get_com(coords):

    centerX = 0.0
    centerY = 0.0
    n = int(len(coords))
    for i in range(n):
        centerX = centerX + coords[i][0]
        centerY = centerY + coords[i][1]
    centerX = centerX/n
    centerY = centerY/n
    center = [centerX, centerY]
    return center


def get_framenum(num):

    framenum = 0
    if num < 10:
        framenum = "00000" + str(num)
    elif num < 100:
        framenum = "0000" + str(num)
    elif num < 1000:
        framenum = "000" + str(num)
    elif num < 10000:
        framenum = "00" + str(num)
    else:
        framenum = "0" + str(num)
    return framenum


def compute_quadratic_programming(Q, p, G, h, A, t):

    # minimize 1/2 XtQX + ptX
    # subject to GX <= h, AX = t

    sol = solvers.qp(Q, p, G, h, A, t)
    return sol['x']


def construct_matrices(orig, offset, M1, M2, M3, M4):

    n = len(orig)
    dim = 4*n
    offset = 10.0

    orig_square = np.square(orig)
    origplusoffset_square = orig_square + 2*offset*orig
    + offset*offset*np.ones(n)
    orig_cube = np.power(orig, 3)
    origplusoffset_cube = orig_cube + 3*orig_square*offset
    + 3*orig*offset*offset + pow(offset, 3.0)*np.ones(n)
    orig_fourth = np.power(orig, 4)
    origplusoffset_fourth = orig_fourth + 4*orig_cube*offset
    + 6*orig_square*offset*offset + 4*orig*pow(offset, 3.0)
    + pow(offset, 4.0)*np.ones(n)

    Q = 2*spmatrix(1.0, range(dim), range(dim))
    p1 = matrix(-2.0*orig)
    p2 = matrix(-2.0*orig_square)
    p3 = matrix(-2.0*orig_cube)
    p4 = matrix(-2.0*orig_fourth)
    p = matrix([p1, p2, p3, p4])

    G = spmatrix(-1.0, range(dim), range(dim))
    h = matrix(0.0, (dim, 1))
    ones = matrix(1.0, (1, n))
    zeros = matrix(0.0, (1, n))
    A1 = matrix([ones, zeros, zeros, zeros])
    A2 = matrix([zeros, ones, zeros, zeros])
    A3 = matrix([zeros, zeros, ones, zeros])
    A4 = matrix([zeros, zeros, zeros, ones])
    A = matrix([[A1], [A2], [A3], [A4]])

    t1 = n*M1
    t2 = n*M2 + n*M1*M1
    t3 = n*pow(M2, 1.5)*M3 + 3*n*M1*M2 + n*M1*M1*M1
    t4 = n*M2*M2*M4 + 4*n*M1*pow(M2, 1.5)*M3 + 6*n*M1*M1*M2 + n*pow(M1, 4)

    t = matrix([t1, t2, t3, t4])
    return [Q, p, G, h, A, t]


def solve_newcoords(coordinates, target_moments):

    # only considering the x coordinates
    # let target_xi = xi + di, di is unknown (i=1,2,..,500, 500 points)
    # the four target moments are known, tMxi (i=1,2,3,4)
    # number of equations 4, number of unknowns, di, is 500
    # this is an underdetermined polynomial system
    # assume a and b are the original x and y coordinates of the points

    n = len(coordinates)
    a = coordinates[:, 0]
    b = coordinates[:, 1]
    offset = 10.0

    [Mx1, Mx2, Mx3, Mx4, My1, My2, My3, My4] = target_moments

    [Q, p, G, h, A, t] = construct_matrices(a, offset, Mx1, Mx2, Mx3, Mx4)
    sol_X = compute_quadratic_programming(Q, p, G, h, A, t)
    print "new x locations:\n", sol_X

    [Q, p, G, h, A, t] = construct_matrices(b, offset, My1, My2, My3, My4)
    sol_Y = compute_quadratic_programming(Q, p, G, h, A, t)
    #print "new y locations:\n", sol_Y

    X1 = list(itertools.chain(*array(sol_X[0:n])))
    X2 = list(itertools.chain(*array(sol_X[n:2*n])))
    X3 = list(itertools.chain(*array(sol_X[2*n:3*n])))
    X4 = list(itertools.chain(*array(sol_X[3*n:])))
    Y1 = list(itertools.chain(*array(sol_Y[0:n])))
    Y2 = list(itertools.chain(*array(sol_Y[n:2*n])))
    Y3 = list(itertools.chain(*array(sol_Y[2*n:3*n])))
    Y4 = list(itertools.chain(*array(sol_Y[3*n:])))

    onethird = 1.0/3.0
    X2 = np.sqrt(X2)
    Y2 = np.sqrt(Y2)
    X3 = np.power(X3, onethird)
    Y3 = np.power(Y3, onethird)
    X4 = np.sqrt(np.sqrt(X4))
    Y4 = np.sqrt(np.sqrt(Y4))

    coords1 = np.column_stack((X1, Y1))
    coords2 = np.column_stack((X2, Y2))
    coords3 = np.column_stack((X3, Y3))
    coords4 = np.column_stack((X4, Y4))

    dist1 = get_dist(coordinates, coords1)
    dist2 = get_dist(coordinates, coords2)
    dist3 = get_dist(coordinates, coords3)
    dist4 = get_dist(coordinates, coords4)

    print dist1, dist2, dist3, dist4

    newcoords = coords1
    dist = dist1
    if dist > dist2:
        newcoords = coords2
        dist = dist2
    if dist > dist3:
        newcoords = coords3
        dist = dist3
    if dist > dist4:
        newcoords = coords4
        dist = dist4

    print get_dist(coordinates, newcoords)
    return newcoords


def my_circle_scatter(axes, x_array, y_array, radius=0.5, **kwargs):

    for x, y in zip(x_array, y_array):
        circle = pylab.Circle((x, y), radius=radius, **kwargs)
        axes.add_patch(circle)
    return True


def drawcoords(coords, newcoords, imgname):

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    my_circle_scatter(ax, zip(*coords)[0], zip(*coords)[1],
                      radius=4.9, alpha=0.7, color='g')
    my_circle_scatter(ax, zip(*newcoords)[0], zip(*newcoords)[1],
                      radius=4.9, alpha=0.5, color='r')
    ax.set_xlim((0, 1000))
    ax.set_ylim((0, 1000))
    plt.savefig(imgname)
    #plt.show()


if __name__ == '__main__':

    step = 3500
    framenum = get_framenum(step)

    targetfilename = sys.argv[1]
    targetCoordinates = np.array(read_coords(targetfilename))
    targetX = targetCoordinates[:, 0]
    targetY = targetCoordinates[:, 1]
    [Mx1, Mx2, Mx3, Mx4] = compute_moments(targetX)
    [My1, My2, My3, My4] = compute_moments(targetY)
    target_moments = [Mx1, Mx2, Mx3, Mx4, My1, My2, My3, My4]

    # distribution to change: calculate deltax and deltay for each point
    filename = sys.argv[2]
    coordinates = np.array(read_coords(filename))
    newcoords = solve_newcoords(coordinates, target_moments)
    np.savetxt("newcoords.txt", newcoords)

    print "===== target ===== "
    print target_moments
    print "===== sol ===== "
    print compute_moments(newcoords[:, 0]) + compute_moments(newcoords[:, 1])

    drawcoords(coordinates, newcoords, "out.png")
