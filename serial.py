#!/usr/bin/env python
# coding: utf-8

# # Dense 3D Face Correspondence

# import os
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import time
import pdb
import numpy as np
import re
import threading
import cv2
import ipyvolume as ipv
import scipy
from math import cos, sin
from scipy import meshgrid, interpolate
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, Delaunay
import numpy as np
from scipy.interpolate import griddata
from collections import defaultdict
#np.warnings.filterwarnings('ignore')
#if not sys.warnoptions:
#    warnings.simplefilter("ignore")


# ## Read each face data

def read_wrl(file_path):
    holder = []
    with open(file_path, "r") as vrml:
        for line in vrml:
            a = line.strip().strip(",").split()
            if len(a) == 3:
                try:
                    holder.append(list(map(float, a)))
                except:
                    pass
    x,y,z = zip(*holder)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    return np.array(holder)


# ## Normalizing faces and Interpolation


def normalize_face(points):
    maxind = np.argmax(points[:,2])
    nosex = points[maxind,0]
    nosey = points[maxind,1]
    nosez = points[maxind,2]
    points = points - np.array([nosex, nosey, nosez])
#     points = points / np.max(points)
    return points


def points2grid(points):
    x1, y1, z1 = map(np.array, zip(*points))
    grid_x, grid_y = np.mgrid[np.amin(x1):np.amax(x1):0.5, np.amin(y1):np.amax(y1):0.5]
    grid_z = griddata((x1, y1), z1, (grid_x, grid_y), method='linear')
    return [grid_x, grid_y, grid_z]


# ## Sparse Correspondence Initialization
# ## Seed points sampling using mean 2D convex hull


def hull72(points, nosex, nosey, nosez):
    newhull = [[nosex, nosey, nosez]]
    for theta in range(0, 360, 5):
        fx = 200 * cos(theta * np.pi / 180)
        fy = 200 * sin(theta * np.pi / 180)
        nearest_point = min(zip(points[:, 0], points[:, 1], points[:, 2]), key=lambda p:(p[0] - fx)**2 + (p[1] - fy)**2)
        newhull.append(nearest_point)
    return newhull


def get_hull(points):
    maxind = np.argmax(points[:,2])
    # coordinates of nose, nosex = x coordinate of nose, similarly for nosey and nosez
    nosex = points[maxind,0]
    nosey = points[maxind,1]
    nosez = points[maxind,2]
    hull = np.array(hull72(points, nosex,nosey,nosez))
    return hull


# ## Delaunay Triangulation

def triangulation(hull):
    points2D = np.vstack([hull[:,0],hull[:,1]]).T
    tri_hull = Delaunay(points2D)
    return tri_hull


# ## Geodesic Patch Extraction



def get_all_patches_from_face(points, hull, triangles):
    from itertools import combinations

    patch_width = 5 * rho
    def distance(x,y,z,x1,y1,z1,x2,y2,z2):
        a = (y2-y1)/(x2-x1)
        b = -1
        c = y2-x2*(y2-y1)/(x2-x1)
        return abs(a*x+b*y+c)/(a**2+b**2)**0.5

    patches = []
    for t1,t2 in combinations(triangles,r=2): #pairwise triangles
        if len(set(t1)&set(t2))==2:           #triangles with a common edge
            patch = []
            a_ind, b_ind = list(set(t1)&set(t2))
            x1, y1, z1 = hull[a_ind,:]
            x2, y2, z2 = hull[b_ind,:]
            for x,y,z in points: #loop over all points to find patch points
                if (x-x1/2-x2/2)**2+(y-y1/2-y2/2)**2<(x1/2-x2/2)**2+(y1/2-y2/2)**2 and distance(x,y,z,x1,y1,z1,x2,y2,z2)<patch_width:
                    patch.append([x,y,z])
            #if patch:
            patches.append(np.array(patch))
    return patches

def get_patches(hull, triangles):
    patches = defaultdict(list) # key = edges, values = a list of extracted patches from all faces along that edge
    for face_index in range(1, len(file_paths)+1):
        all_patches = get_all_patches_from_face(face_points["face"+str(face_index)], hull, triangles)
        #print(len(all_patches))
        # the patches are organised in following way because the original get_patches function was modified after the whole serial code was written
        for edge_index in range(len(all_patches)):
            patches["edge" + str(edge_index)].append(all_patches[edge_index-1])
    return patches


 ## Keypoint Extraction

# takes in a point and the patch it belongs to and decides whether it is a keypoint (ratio of largest two eigenvalues on the covariance matrix of its local surface) or not
def is_keypoint(point, points):
    threshold = 7 * rho
    nhood = points[(np.sum(np.square(points-point),axis=1)) < threshold**2]
    try:
        nhood = (nhood - np.min(nhood, axis=0)) / (np.max(nhood, axis=0) - np.min(nhood, axis=0))
        covmat = np.cov(nhood)
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(covmat)))
        ratio = eigvals[-1]/(eigvals[-2]+0.0001)
        return ratio>30 #eigen_ratio_threshold #/ 5
    except Exception as e:
        return False


def get_keypoints(patches):
    keypoints = {} # key = edge, value = a list of keypoints extracted from the patches along that edge across all faces
    for edge_index in range(1, len(patches)+1):
        edge_patches = patches["edge" + str(edge_index)]
        edge_keypoints = []
        for patch in edge_patches:
            #print(patch.shape)
            if patch.shape[0]:
                patch_keypoints = patch[np.apply_along_axis(is_keypoint, 1, patch, patch)] # keypoints in `patch`
            else:
                patch_keypoints = []
            edge_keypoints.append(patch_keypoints)
        keypoints["edge" + str(edge_index)] = edge_keypoints
    return keypoints

# ## Feature Extraction


def get_normal(x, y, grid_x, grid_y, grid_z):
    '''
      3
    1   2
      4
    x, y are coordinates of the point for which the normal has to be calculated
    '''
    i = (x - grid_x[0, 0]) / (grid_x[1, 0] - grid_x[0, 0])
    j = (y - grid_y[0, 0]) / (grid_y[0, 1] - grid_y[0, 0])
    i,j = int(round(i)), int(round(j))
    if (not 0 <= i < grid_x.shape[0]-1) or (not 0 <= j < grid_y.shape[1]-1):
        warnings.warn("out of bounds error")
        #pdb.set_trace()
        return "None"
    point1 = (grid_x[i-1, j], grid_y[i-1, j], grid_z[i-1, j])
    point2 = (grid_x[i+1, j], grid_y[i+1, j], grid_z[i+1, j])
    point3 = (grid_x[i, j-1], grid_y[i, j-1], grid_z[i, j-1])
    point4 = (grid_x[i, j+1], grid_y[i, j+1], grid_z[i, j+1])
    a1, a2, a3 = [point2[x] - point1[x] for x in range(3)]
    b1, b2, b3 = [point3[x] - point4[x] for x in range(3)]
    normal = np.array([a3*b2, a1*b3, -a1*b2])
    return normal/np.linalg.norm(normal)



# moments = cv2.moments(patch2[:, :2])
# central_moments = [moments[key] for key in moments.keys() if key[:2] == "mu"]
# central_moments = np.array(central_moments)
# central_moments


def get_keypoint_features(keypoints, face_index):
    feature_list = [] # a list to store extracted features of each keypoint
    final_keypoints = [] # remove unwanted keypoints, like the ones on edges etc
    for point in keypoints:
        point_features = []
        x, y, z = point
        points = face_points["face" + str(face_index)]
        grid_x, grid_y, grid_z = grid_data["face" + str(face_index)]
        threshold = 5 * rho
        nhood = points[(np.sum(np.square(points-point), axis=1)) < threshold**2]
        xy_hu_moments = cv2.HuMoments(cv2.moments(nhood[:, :2])).flatten()
        yz_hu_moments = cv2.HuMoments(cv2.moments(nhood[:, 1:])).flatten()
        xz_hu_moments = cv2.HuMoments(cv2.moments(nhood[:, ::2])).flatten()
        hu_moments = np.concatenate([xy_hu_moments, yz_hu_moments, xz_hu_moments])
        #print(hu_moments)
        #i = (x - grid_x[0, 0]) / (grid_x[1, 0] - grid_x[0, 0])
        #j = (y - grid_y[0, 0]) / (grid_y[0, 1] - grid_y[0, 0])
        #i, j = int(round(i)), int(round(j))
        #start_i, start_j = i - int(5 * rho / (grid_x[1, 0] - grid_x[0, 0])), j - int(5 * rho / (grid_y[0, 1] - grid_y[0, 0]))
        #end_i, end_j = i + int(5 * rho / (grid_x[1, 0] - grid_x[0, 0])), j + int(5 * rho / (grid_y[0, 1] - grid_y[0, 0]))
        #nhood = points[start_i: end_i, start_j: end_j]
        #nhood_x = grid_x[start_i:end_i, start_j:end_j]
        #nhood_y = grid_y[start_i:end_i, start_j:end_j]
        #nhood_z = grid_z[start_i:end_i, start_j:end_j]
        normal = get_normal(x, y, grid_x, grid_y, grid_z)
        if normal == "None": # array comparision raises ambiguity error, so None passed as string
            continue
        final_keypoints.append(point)
        point_features.extend(np.array([x, y, z])) # spatial location
        point_features.extend(normal)
        point_features.extend(hu_moments)
        point_features = np.array(point_features)

        feature_list.append(point_features)
    final_keypoints = np.array(final_keypoints)
    return final_keypoints, feature_list


# In[104]:


def get_features(keypoints):
    features = {} # key = edge + edge_index, value = list of features for each keypoint across all the faces
    for edge_index in range(1, len(keypoints)+1):
        edgewise_keypoint_features = [] # store features of keypoints for a given edge_index across all faces
        for face_index in range(1, len(file_paths)+1):
            try:
                edge_keypoints = keypoints["edge" + str(edge_index)][face_index-1]
                final_keypoints, keypoint_features = get_keypoint_features(edge_keypoints, face_index)
                keypoints["edge" + str(edge_index)][face_index-1] = final_keypoints # update the keypoint, remove unwanted keypoints like those on the edge etc
            except:
                keypoint_features = []
            edgewise_keypoint_features.append(keypoint_features)
        features["edge" + str(edge_index)] = edgewise_keypoint_features
    return features

# ## Keypoint matching

# In[97]:


def get_keypoint_under_2rho(keypoints, point):
    """return the index of the keypoint in `keypoints` which is closest to `point` if that distance is less than 2 * rho, else return None"""
    try:
        distance = np.sqrt(np.sum(np.square(keypoints-point), axis=1))
        if (distance < 3*rho).any():
            min_dist_index = np.argmin(distance)
            return min_dist_index
    except Exception as e: # keypoints is [], gotta return None
        pass
    return None

def get_matching_keypoints(edge_keypoints, edge_features, edge_index):
    # check if a bunch of keypoints across the patches (across all faces) are withing 2*rho
    # first get all the keypoints in a list
    matching_keypoints_list = []
    for face_index1 in range(len(edge_keypoints)): # take a patch along the edge among the faces
        for point_index, point in enumerate(edge_keypoints[face_index1]): # take a keypoint in that patch, we have to find corresponding keypoints in each other patche along this edge
            matched_keypoint_indices = [] # to store indices of matched keypoints across the patches
            for face_index2 in range(len(edge_keypoints)): # find if matching keypoints exist across the patches along that edge across all faces
                if face_index2 == face_index1:
                    matched_keypoint_indices.append(point_index)
                    continue
                matched_keypoint = get_keypoint_under_2rho(edge_keypoints[face_index2], point)
                if matched_keypoint:
                    #if edge_index == 36: pdb.set_trace()I#
                    matched_keypoint_indices.append(matched_keypoint)
                else: # no keypoint was matched in the above patch (face_index2), gotta start search on other keypoint from face_index1
                    break

            if len(matched_keypoint_indices) == len(edge_keypoints): # there's a corresponding keypoint for each patch across all faces
                 matching_keypoints_list.append(matched_keypoint_indices)
    if len(matching_keypoints_list) == 0:
        return []
    # time we have those keypoints which are in vicinity of 2*rho, let's compute euclidean distance of their feature vectors
    final_matched_keypoints = []
    for matched_keypoints in matching_keypoints_list: # select first list of matching keypoints
        # get the indices, get their corresponding features, compute euclidean distance
        try:
            features = np.array([edge_features[face_index][idx] for face_index, idx in zip(range(len(edge_features)), matched_keypoints)])
            euc_dist_under_kq = lambda feature, features: np.sqrt(np.sum(np.square(features - feature), axis=1)) < Kq
            if np.apply_along_axis(euc_dist_under_kq, 1, features, features).all() == True:
                # we have got a set of matching keypoints, get their mean coordinates
                matched_coords = [edge_keypoints[face_index][idx] for face_index, idx in zip(range(len(edge_features)), matched_keypoints)]
                final_matched_keypoints.append(np.mean(matched_coords, axis=0))
        except Exception as e:
            print(e)
            pdb.set_trace()
    return final_matched_keypoints


# In[98]:


# those keypoints which are in vicinity of 2*rho are considered for matching
# matching is done using constrained nearest neighbour
# choose an edge, select a keypoint, find out keypoints on corresponding patches on other faces within a vicinity of 2*rho,
# get euclidean distance in features among all possible pair wise combinations, if the distances come out to be less than Kp are added to the global set of correspondences
def keypoint_matching_process(keypoints, features):
    final_mean_keypoints = []
    for edge_index in range(1, len(keypoints)):
        edge_keypoints = keypoints["edge" + str(edge_index)]
        edge_features = features["edge" + str(edge_index)]
        matched_keypoints = get_matching_keypoints(edge_keypoints, edge_features, edge_index)
        if len(matched_keypoints) == 0:
            continue
        #print(matched_keypoints)
        final_mean_keypoints.extend(matched_keypoints)
    #final_mean_keypoints = list(set(final_mean_keypoints))

    final_mean_keypoints = np.array(final_mean_keypoints)
    final_mean_keypoints = np.unique(final_mean_keypoints, axis=0)
    return final_mean_keypoints


# THRESHOLDS
rho = 0.5
eigen_ratio_threshold = 5000
Kq = 10

file_paths = {
    "path1": "F0001/F0001_AN01WH_F3D.wrl",
    "path2": "F0001/F0001_AN02WH_F3D.wrl",
    "path3": "F0001/F0001_AN03WH_F3D.wrl",
    "path4": "F0001/F0001_AN04WH_F3D.wrl",
    "path5": "F0001/F0001_DI01WH_F3D.wrl",
    "path6": "F0001/F0001_DI02WH_F3D.wrl",
    "path7": "F0001/F0001_DI03WH_F3D.wrl",
    "path8": "F0001/F0001_DI04WH_F3D.wrl",
}
print("Reading face data of %d files............" % len(file_paths), end="", flush=True)
face_points = {} # key = face+index, value = extracted face data
for i in range(1, len(file_paths)+1):
    face_points["face" + str(i)] = read_wrl(file_paths["path" + str(i)])
print("Done")
# normalizing the faces and interpolating them across a grid
print("Prepapring normalized face data and grid data............ ", end="", flush=True)
grid_data = {}
for i in range(1, len(file_paths)+1):
    # normalization
    face_points["face" + str(i)] = normalize_face(face_points["face" + str(i)])
    # grid interpolation of the face data
    grid_data["face" + str(i)] = points2grid(face_points["face" + str(i)])
print("Done")
print("Extracting mean 2D Convex hull...........", end="", flush=True)
hull = np.zeros([73, 3])
for i in range(1, len(file_paths)+1):
    hull += get_hull(face_points["face" + str(i)])
hull = hull / len(file_paths)
print("Done")
print("Starting the iterative process............")

#tri_hull = triangulation(hull)
#patches = get_patches(hull)
#keypoints = get_keypoints(patches)
#features = get_features(keypoints)
#final_mean_keypoints = keypoint_matching_process(keypoints, features)
#print(final_mean_keypoints)
#updated_hull = np.concatenate((hull, final_mean_keypoints), axis=0)


# Start correspondence densification loop
num_iterations = 10
correspondence_set = hull
for iteration in range(num_iterations):
    print("\n\nStarting iteration: ", iteration)
    t1 = time.time()
    print("Starting Delaunay triangulation............", end="", flush=True)
    tri_hull = triangulation(correspondence_set)
    print("Done | time taken: %0.4f seconds" % (time.time() - t1))

    t2 = time.time()
    print("Starting geodesic patch extraction............", end="", flush=True)
    patches = get_patches(correspondence_set, tri_hull.simplices)
    print("Done | time taken: %0.4f seconds" % (time.time() - t2))

    t3 = time.time()
    print("Starting keypoint extraction............", end="", flush=True)
    keypoints = get_keypoints(patches)
    print("Done | time taken: %0.4f seconds" % (time.time() - t3))

    t4 = time.time()
    print("Starting feature extraction............", end="", flush=True)
    features = get_features(keypoints)
    print("Done | time taken: %0.4f seconds" % (time.time() - t4))

    t5 = time.time()
    print("Starting keypoint matching............", end="", flush=True)
    final_mean_keypoints = keypoint_matching_process(keypoints, features)
    print("Done | time taken: %0.4f seconds" % (time.time() - t5))

    print("Total new correspondences found: ", len(final_mean_keypoints))
    print("Updating correspondence set...")
    correspondence_set = np.concatenate((correspondence_set, final_mean_keypoints), axis=0)
    correspondence_set = np.unique(correspondence_set, axis=0)
    print("Iteration completed in %0.4f seconds" % (time.time() - t1))



