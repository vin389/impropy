from math import cos, sin, tan, acos, asin, atan2, sqrt, pi
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from numba import jit
import time

def rvecTvecFromR44(r44): 
    """
    Returns the rvec and tvec of the camera

    Parameters
    ----------
    r44 : TYPE np.array((4,4),dtype=float)
        The 4-by-4 form of camera extrinsic parameters
    Returns
    -------
    TYPE: tuple (np.array((3,1),dtype=float)
    Returns the rvec and tvec of the camera ([0] is rvec; [1] is tvec.)

    """
    rvec, rvecjoc = cv.Rodrigues(r44[0:3,0:3])
    tvec = r44[0:3,3]
    return rvec, tvec

def r44ToRvecTvec(r44):  # the same as rvecTvecFromR44(r44)
    """
    Returns the rvec and tvec of the camera
    (the same as rvecTvecFromR44)
    
    Parameters
    ----------
    r44 : TYPE np.array((4,4),dtype=float)
        The 4-by-4 form of camera extrinsic parameters
    Returns
    -------
    TYPE: tuple (np.array((3,1),dtype=float)
    Returns the rvec and tvec of the camera ([0] is rvec; [1] is tvec.)
    
    """
    rvec, rvecjoc = cv.Rodrigues(r44[0:3,0:3])
    tvec = r44[0:3,3]
    return rvec, tvec

def r44FromRvecTvec(rvec, tvec): 
    """
    Returns the 4-by-4 coordinate transformation matrix of the camera

    Parameters
    ----------
    rvec: TYPE np.array(3, dtype=float)
    tvec: TYPE np.array(3, dtype=float)
    
    r44 : TYPE np.array((4,4),dtype=float)
        The 4-by-4 form of camera extrinsic parameters
    Returns
    -------
    TYPE: tuple ([0]: np.array(3,dtype=float, [1]: np.array(3,dtype=float)))
    the 4-by-4 coordinate transformation matrix of the camera

    """
    r44 = np.eye(4, dtype=np.float)
    r33, r33joc = cv.Rodrigues(rvec)
    r44[0:3,0:3] = r33.copy()
    r44[0:3,3] = tvec.reshape(3)
    return r44

def rvecTvecToR44(rvec, tvec): # the same as r44FromRvecTvec
    """
    Returns the 4-by-4 coordinate transformation matrix of the camera

    Parameters
    ----------
    rvec: TYPE np.array(3, dtype=float)
    tvec: TYPE np.array(3, dtype=float)
    
    r44 : TYPE np.array((4,4),dtype=float)
        The 4-by-4 form of camera extrinsic parameters
    Returns
    -------
    TYPE: tuple ([0]: np.array(3,dtype=float, [1]: np.array(3,dtype=float)))
    the 4-by-4 coordinate transformation matrix of the camera

    """
    r44 = np.eye(4, dtype=np.float)
    r33, r33joc = cv.Rodrigues(rvec)
    r44[0:3,0:3] = r33.copy()
    r44[0:3,3] = tvec.copy()
    return r44

def extrinsicR44ByCamposAndAim(campos, aim):
    """
    Calculates the 4-by-4 matrix form of extrinsic parameters of a camera according to camera position and a point it aims at.
    Considering the world coordinate X-Y-Z where Z is upward, 
    starting from an initial camera orientation (x,y,z) which is (X,-Z,Y), that y is downward (-Z), 
    rotates the camera so that it aims a specified point (aim)
    This function guarantee the camera axis x is always on world plane XY (i.e., x has no Z components)
    Example:
        campos = np.array([ -100, -400, 10],dtype=float)
        aim = np.array([0, -50, 100],dtype=float)
        r44Cam = extrinsicR44ByCamposAndAim(campos,aim)
        # r44Cam would be 
        # np.array([[ 0.961, -0.275,  0.000, -1.374],
        #           [ 0.066,  0.231, -0.971,  108.6],
        #           [ 0.267,  0.933,  0.240,  397.6],
        #           [ 0.000,  0.000,  0.000,  1.000]])
        
    Parameters
    ----------
    campos: TYPE np.array((3,3),dtype=float)
        camera position in the world coordinate 
    aim: TYPE np.array((3,3),dtype=float)
        the aim that the camera is aims at

    Returns
    -------
    TYPE: np.array((4,4),dtype=float)
        the 4-by-4 matrix form of the extrinsic parameters
    """
    # camera vector (extrinsic)
    vz_cam = aim - campos
    vy_cam = np.array([0,0,-1], dtype=np.float64)
    vz_cam = vz_cam / np.linalg.norm(vz_cam)
    vx_cam = np.cross(vy_cam, vz_cam)
    vy_cam = np.cross(vz_cam, vx_cam)
    vx_cam = vx_cam / np.linalg.norm(vx_cam)
    vy_cam = vy_cam / np.linalg.norm(vy_cam)
    vz_cam = vz_cam / np.linalg.norm(vz_cam)
    r44inv = np.eye(4, dtype=np.float64)
    r44inv[0:3, 0] = vx_cam[0:3]
    r44inv[0:3, 1] = vy_cam[0:3]
    r44inv[0:3, 2] = vz_cam[0:3]
    r44inv[0:3, 3] = campos[0:3]
    r44 = np.linalg.inv(r44inv)
    return r44

def extrinsicR44ByCamposYawPitch(campos, yaw, pitch):
    """
    Calculates the 4-by-4 matrix form of extrinsic parameters of a camera according to camera yaw and pitch.
    Considering the world coordinate X-Y-Z where Z is upward, 
    starting from an initial camera orientation (x,y,z) which is (X,-Z,Y), that y is downward (-Z), 
    rotates the camera y axis (yaw, right-hand rule) then camera x axis (pitch, right-hand rule) in degrees.
    This function guarantee the camera axis x is always on world plane XY (i.e., x has no Z components)
    Example:
        campos = np.array([ -100, -400, 10],dtype=float)
        yaw = 15.945395900922847; pitch = 13.887799644071938;
        r44Cam = extrinsicR44ByCamposYawPitch(campos, yaw, pitch)
        # r44Cam would be 
        # np.array([[ 0.961, -0.275,  0.000, -1.374],
        #           [ 0.066,  0.231, -0.971,  108.6],
        #           [ 0.267,  0.933,  0.240,  397.6],
        #           [ 0.000,  0.000,  0.000,  1.000]])
        
    Parameters
    ----------
    campos: TYPE np.array((3,3),dtype=float)
        camera position in the world coordinate 
    yaw: TYPE float
        camera yaw along y axis (right-hand rule) (in degree), typically clockwise positive
    pitch: TYPE float
        camera pitch along x axis (right-hand rule) (in degree), typically upward positive

    Returns
    -------
    TYPE: np.array((4,4),dtype=float)
        the 4-by-4 matrix form of the extrinsic parameters
    """
    # camera vector (extrinsic)
    vxx = cos(yaw * pi / 180.)
    vxy = -sin(yaw * pi / 180.)
    vxz = 0
    vx_cam = np.array([vxx, vxy, vxz], dtype=np.float64)
    vzx = sin(yaw * pi / 180.) * cos(pitch * pi / 180.)
    vzy = cos(yaw * pi / 180) * cos(pitch * pi / 180.)
    vzz = sin(pitch * pi / 180.)
    vz_cam = np.array([vzx, vzy, vzz], dtype=np.float64)
    vy_cam = np.cross(vz_cam, vx_cam)
    r44inv = np.eye(4, dtype=np.float64)
    r44inv[0:3, 0] = vx_cam[0:3]
    r44inv[0:3, 1] = vy_cam[0:3]
    r44inv[0:3, 2] = vz_cam[0:3]
    r44inv[0:3, 3] = campos[0:3]
    r44 = np.linalg.inv(r44inv)
    return r44

#@jit(nopython=True)
def extrinsicR44ByCamposYawPitch2(campos, yaw, pitch):
    phi = pitch * pi / 180.
    theta = yaw * pi / 180
    r44inv = np.eye(4, dtype=np.float64)
    r44inv[0,0] = cos(theta)
    r44inv[1,0] = -sin(theta)
    r44inv[2,0] = 0
    r44inv[0,1] = sin(phi) * sin(theta)
    r44inv[1,1] = sin(phi) * cos(theta)
    r44inv[2,1] = -cos(phi)
    r44inv[0,2] = cos(phi) * sin(theta)
    r44inv[1,2] = cos(phi) * cos(theta)
    r44inv[2,2] = sin(phi)
    r44inv[0,3] = campos[0]        
    r44inv[1,3] = campos[1]        
    r44inv[2,3] = campos[2]        
    r44 = np.linalg.inv(r44inv)
    return r44
    

def bladePointByThetaAndDeflection(r_blade, r44_blade, theta, deflection): 
    """
    calculates the 3D coordinates of blade point 
    Example: 
        # the radius of the point on the blade is 55 meters
        r_blade = 55.0; 
        # blade faces towards -Y of global coord. (i.e., blade y is 0,-1,0)
        r44_blade = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]],dtype=float)
        # blade rotates to the highest point
        theta = 90; 
        # deflection is 0.3 m along blade z axis
        deflection = 0.3
        # get the coordinate of peak (peakPoint)
        peakPoint = bladePointByThetaAndDeflection(r_blade, r44_blade, theta, deflection)
    
    Parameters
    ----------
    r_blade : TYPE float
        radius of the blade point
    r44_blade : TYPE numpy.ndarray ((4,4), dtype=np.float)
        the extrinsic parameters of the blade in 4-by-4 matrix form
    theta : TYPE float 
        the angle (in degree) of the blade on the blade axes x and y (note: the blade y axis could be commonly downward. Think carefully about the blade axes according to the r44_blade)
    deflection : TYPE float
        the deflection of the blade along blade axis z 
    
    Returns
    -------
    TYPE (4, dtype=np.float) 
    Returns the 3D coordinate of the point (homogeneous coordinate).
    """
    bladePointLocal = np.array([
                        r_blade * cos(theta / (180./pi)), \
                        deflection, 
                        r_blade * sin(theta / (180./pi)), 1], \
                        dtype=np.float64)
    r44inv_blade = np.linalg.inv(r44_blade)  
    bladePointGlobal = np.matmul(r44inv_blade, bladePointLocal.transpose())
    bladePointGlobal /= bladePointGlobal[3]
    return bladePointGlobal

def bladeImgPointByThetaAndDeflection(theta, deflection, r_blade, r44_blade, cmat, dvec, r44_cam):
    """
    calculates the image coordinates of blade point 

    Parameters
    ----------
    theta : TYPE float 
        the angle (in degree) of the blade on the blade axes x and y (note: the blade y axis could be commonly downward. Think carefully about the blade axes according to the r44_blade)
    deflection : TYPE float
        the deflection of the blade along blade axis z 
    r_blade : TYPE float 
        radius of the blade point
    r44_blade : TYPE numpy.ndarray ((4,4), dtype=np.float)
        the extrinsic parameters of the blade in 4-by-4 matrix form
    cmat : TYPE numpy.ndarray((3,3), dtype=np.float)
        camera matrix 
    dvec : TYPE numpy.ndarray(n, dtype=np.float)
        distortion vector
    r44_cam : TYPE numpy.ndarray((4,4),dtype=np.float)
        extrinsic parameters

    Returns
    -------
    TYPE numpy array (2, dtype=np.float) 
    Returns the image coordinate of the point .
    """
    bladeWorldPoint = bladePointByThetaAndDeflection(r_blade, r44_blade, theta, deflection)
    rvec, tvec = rvecTvecFromR44(r44_cam)
    bladeImagePoint, jacob = cv.projectPoints(bladeWorldPoint[0:3],                                               
                                              rvec, tvec, cmat, dvec)    
    bladeImagePoint = bladeImagePoint.reshape(2)
    return bladeImagePoint

def funBladeImgPointByThetaAndDeflection(x, r_blade, r44_blade, cmat, dvec, r44_cam, imgPoint):
    theta = x[0]
    deflection = x[1]   
    bladeImagePoint = bladeImgPointByThetaAndDeflection(theta, deflection, r_blade, r44_blade, cmat, dvec, r44_cam)   
    bladeImagePoint -= imgPoint 
    return bladeImagePoint

def bladeThetaAndDeflectionByImgPoint(bladeImagePoint, r_blade, r44_blade, cmat, dvec, r44_cam):
    """
    bladeThetaAndDeflectionByImgPoint()
    This function estimates the rotatinal angle (theta), the deflection, and the 
    numerical condition number of a wind turbine blade, given the image point, 
    the distance between the target and the rotational axis (the radius), the 
    extrinsic parameters (a 4-by-4 matrix) of the turbine, the intrinsic and 
    extrinsic parameters of the camera). 
    
    Parameters
    ----------
    bladeImagePoint : TYPE numpy array (2, dtype=float) 
        the image coordinate of the point 
    r_blade : TYPE float
        the radius (distance between of target and rotational axis)
    r44_blade : TYPE numpy array ((4, 4), dtype=float)
        the extrinsic parameters of the wind turbine
    cmat : TYPE numpy array ((3, 3), dtype=float)
        the camera matrix
    dvec : TYPE numpy array((n), dtype=float)
        the distortion vector (n = 4, 5, 8, .... See OpenCV manual for details)
    r44_cam : TYPE numpy.ndarray((4,4),dtype=np.float)
        extrinsic parameters of the camera
    
    Returns
    -------
    TYPE tuple 
        angle: TYPE float
               the angle (in degree) of the blade on the plan axes x and y (note: 
               the blade y axis could be commonly downward. Think carefully about 
               the blade axes according to the given r44_blade
        deflection: TYPE float
               the deflection of the blade along blade axis z+
        condition: TYPE float
               the condition number of the jacobi matrix (of the numerical system)
               The larger the condition number is, the more sensitive (unstable) 
               the system could be. (That is, small image coordinate change can 
               induce large estimated displacement.)
    """ 
    minCost = 1e30
    bestTheta = -1;
    bestDeflection = -1;
    bestRes =[];
    for theta_i in range(36):
        initTheta = theta_i * 10.0 
        x0 = np.array((initTheta, 0),dtype=float)
        lbound = np.array([  0.0, -r_blade * 0.2])
        ubound = np.array([360.0, +r_blade * 0.2])
        bounds = (lbound, ubound)
        res_lsq = least_squares(funBladeImgPointByThetaAndDeflection, x0, \
            bounds= bounds, 
            args=(r_blade, r44_blade, cmat, dvec, r44_cam, bladeImagePoint))
        if (res_lsq.cost < minCost):
            minCost = res_lsq.cost
            bestTheta = res_lsq.x[0]
            bestDeflection = res_lsq.x[1]
            bestRes = res_lsq
#        print(res_lsq)
#        print('-----------------------')
    eigs = np.linalg.eig(bestRes.jac)
    condition = max(abs(eigs[0])) / min(abs(eigs[0]))
    return bestTheta, bestDeflection, condition
        
#

               
def bladeThetaAndDeflectionByImgPoint_fastTrial(bladeImagePoint, r_blade, r44_tur_wld, cmat, dvec, r44_cam_wld):
    """
    bladeThetaAndDeflectionByImgPoint_fastTrial()
    This is a fast version of bladeThetaAndDeflectionByImgPoint(). Normally these
    two function returns very close result when the condition number is not large.
    
    The function bladeThetaAndDeflectionByImgPoint() calls least_squares() for 
    many times, and picks the best one.
    
    The function bladeThetaAndDeflectionByImgPoint_fastTrial() calls 
    least_squares() only once, which is supposed to be faster. 
    
    When the condition number is large, none of these two function guarangees 
    good convergence. 
    
    A condition number which is < 100 should be safe for convergence. 
    
    """
#    minCost = 1e30
    minCost = 1e30
    bestInitTheta = -1;
    bestInitDeflection = -1;
    n360 = 10
    for theta_i in range(n360):
        initTheta = theta_i * (360. / n360) + 0.56789
        initDefln = 0.0
        x0 = np.array((initTheta, initDefln),dtype=float)
        projErr = funBladeImgPointByThetaAndDeflection( \
                  x0, r_blade, r44_tur_wld, cmat, dvec, r44_cam_wld, \
                  bladeImagePoint)
        projErrNorm = np.linalg.norm(projErr)
        if (projErrNorm < minCost):
            minCost = projErrNorm
            bestInitTheta = x0[0]
            bestInitDeflection = x0[1]
    x0 = np.array((bestInitTheta, bestInitDeflection),dtype=float)
    lbound = np.array([-90.0, -r_blade * 0.2])
    ubound = np.array([450.0, +r_blade * 0.2])
    bounds = (lbound, ubound)
    res_lsq = least_squares(funBladeImgPointByThetaAndDeflection, x0, \
        bounds= bounds, 
        args=(r_blade, r44_tur_wld, cmat, dvec, r44_cam_wld, bladeImagePoint))
    eigs = np.linalg.eig(res_lsq.jac)
    condition = max(abs(eigs[0])) / min(abs(eigs[0]))
    bestTheta = res_lsq.x[0] 
    bestTheta = (bestTheta + 720.) % 360.
    bestDeflection = res_lsq.x[1]
    return bestTheta, bestDeflection, condition
    


# wind turbine
def test_blade_55():
    bladeCenter = np.array([0, -50, 100], dtype=np.float64)
    bladeAim = bladeCenter + np.array([0, -100, 1], dtype=np.float64)
    r44_blade_wld = extrinsicR44ByCamposAndAim(bladeCenter, bladeAim)
    rvec_blade_wld, tvec_blade_wld = rvecTvecFromR44(r44_blade_wld)
    r44_wld_blade = np.linalg.inv(r44_blade_wld)
    r_blade = 55.0
    
    # camera extrinsic 
    camPos = np.array([ -100, -400, 10], dtype = np.float64)
    camAim = bladeCenter
    r44_cam_wld = extrinsicR44ByCamposAndAim(camPos, camAim)
    rvec_cam_wld, rvecjoc = cv.Rodrigues(r44_cam_wld[0:3,0:3])
    tvec_cam_wld = r44[0:3,3]
    
    # synthetic camera intrinsic
    imgh = 1080; imgw = 1920;
    fx = fy = 3000
    cx = (imgw - 1) / 2.0; cy = (imgh - 1) / 2.0
    cmat = np.eye(3, dtype=np.float64)
    cmat[0,0] = fx; cmat[1,1] = fy; cmat[0,2] = cx; cmat[1,2] = cy;
    dvec = np.zeros((1,8), dtype=np.float64)
    
    
    correctTheta = np.zeros(360, dtype=float)
    correctDeflection = np.zeros(360, dtype=float)
    calcTheta = np.zeros(360, dtype=float)
    calcDeflection = np.zeros(360, dtype=float)
    xis = np.zeros(360, dtype=float)
    yis = np.zeros(360, dtype=float)
    calcCondition = np.zeros(360, dtype=float)
    
    # trajectory to image
    img = np.zeros((imgh, imgw), dtype=np.uint8)
    steps = range(360)
    for i in range(360):
    #    bladePeak = np.array([r_blade * cos(i / (180./pi)), \
    #                r_blade * sin(i / (180./pi)), 0, 1], dtype=np.float64)
    #    bladePeakGlobal = np.matmul(r44inv_blade, bladePeak.transpose())
        theta = 1.0 * i
        deflection = 5.0 * sin(20.* i / 57.295779)
        bladeImagePoint = bladeImgPointByThetaAndDeflection( \
            theta, deflection, r_blade, r44_blade, cmat, dvec, r44_cam_wld)
        yi = int(bladeImagePoint[1] + 0.5)
        xi = int(bladeImagePoint[0] + 0.5)
        
        calcTheta[i], calcDeflection[i], calcCondition[i] = \
            bladeThetaAndDeflectionByImgPoint(
                bladeImagePoint, \
                r_blade, r44_blade, cmat, dvec, r44_cam_wld)
        correctTheta[i] = i
        correctDeflection[i] = deflection
        xis[i] = xi
        yis[i] = yi
        if yi >= 0 and yi < imgh and xi >= 0 and xi < imgw:
            img[yi, xi] = 255
    deflPlot = plt.figure(1)
    plt.plot(steps, calcDeflection)
    thetPlot = plt.figure(2)
    plt.plot(steps, calcTheta)
    imgPlot = plt.figure(3)
    plt.imshow(img, cmap='gray')
    condPlot = plt.figure(4)
    plt.plot(range(360), calcCondition)

def funMinDistBetweenPoints2dAndManyPoints2dByCamposYawPitch(xyzyp, cmat, dvec, ptsMany3d, xis):
    nPoint = int(xis.size / 2 + 0.5)
    nMany = int(ptsMany3d.size / 3 + 0.5)
    campos = xyzyp[0:3].copy()
    yaw = xyzyp[3]; pitch = xyzyp[4]
    r44_cam_turb = extrinsicR44ByCamposYawPitch(campos, yaw, pitch)
    rvec_ct, tvec_ct = rvecTvecFromR44(r44_cam_turb)
    xis_many_prj, tmp = cv.projectPoints(ptsMany3d,
        rvec_ct, tvec_ct, cmat, dvec)   
    res = findMinDistBetweenPoints2dAndManyPoints2d(xis, xis_many_prj)
    return res

def funMinDistBetweenPoints2dAndManyPoints2dByRvecTvec(xRvecTvec, cmat, dvec, ptsMany3d, xis):
    nPoint = int(xis.size / 2 + 0.5)
    nMany = int(ptsMany3d.size / 3 + 0.5)
    xis_many_prj, tmp = cv.projectPoints(ptsMany3d.copy(), \
        xRvecTvec[0:3].reshape(3,1), xRvecTvec[3:6].reshape(3,1), cmat, dvec)  
    res = np.zeros(nPoint, dtype=np.float)
    findMinDistBetweenPoints2dAndManyPoints2d(xis, xis_many_prj, res)
    return res

@jit(nopython=True)
def findMinDistBetweenPoints2dAndManyPoints2d(pts, ptsMany, res):
    # data reshape
    nPoint = int(pts.size / 2 + .5)
    nMany = int(ptsMany.size / 2 + .5)
    pts = pts.reshape(nPoint, 2)
    ptsMany = ptsMany.reshape(nMany, 2)
    # find minimum distance between each pts and ptsMan
    for i in range(nPoint):
        minDistSq = 1.e37
        for j in range(nMany):
            dx = pts[i,0] - ptsMany[j,0]
            dy = pts[i,1] - ptsMany[j,1]
            distSq = dx * dx + dy * dy
            if distSq < minDistSq:
                minDistSq = distSq
        res[i] = minDistSq                
                 
"""
# wind turbine
# This function demonstrates how to find the deflection of a wind turbine 
# blade by calling the following function(s):
#   bladeThetaAndDeflectionByImgPoint(), or 
#   bladeThetaAndDeflectionByImgPoint_fastTrial() (which is a faster version)
# The scenario is as follows.
# The blade center (the origin of the wind turbine coordinate system) is 
# [0, -50, 100] (meters)
# The turbine aims the point [0, -100, 1] (related to the blade center, i.e., 
#  aiming the point [0, -150, 101])
# the distance between target and axis (radius) is 55 meters
# The camera is at [-100, -400, -10], aiming the turbine center 
# The focal lengths fx and fy both are 3000 pixels. Distortions are zeros. 
# Assuming 360 images were (virtually) taken, and the image coordinates are 
# (accurately) calculated. 
# The 
               
"""

def test_blade_55_fastTrial():
    # blade extrinsic parameters and radius 
    turbineCenter = np.array([0, -50, 100], dtype=np.float64)
    r44_tur_wld = extrinsicR44ByCamposYawPitch2( \
        campos=turbineCenter, yaw=180., pitch=90 + 0.0 * 180 / pi)   
    r44_wld_tur = np.linalg.inv(r44_tur_wld)                                                        
    r_blade = 55.0
    
    # camera extrinsic 
    camPos = np.array([ -100, -400, 10], dtype = np.float64)
    camAim = turbineCenter
    r44_cam_wld = extrinsicR44ByCamposAndAim(camPos, camAim)
    rvec_cam_wld, rvecjoc = cv.Rodrigues(r44_cam_wld[0:3,0:3])
    tvec = r44_cam_wld[0:3,3]
    
    # synthetic camera intrinsic
    imgh = 1080; imgw = 1920;
    fx = fy = 3000
    cx = (imgw - 1) / 2.0; cy = (imgh - 1) / 2.0
    cmat = np.eye(3, dtype=np.float64)
    cmat[0,0] = fx; cmat[1,1] = fy; cmat[0,2] = cx; cmat[1,2] = cy;
    dvec = np.zeros((1,8), dtype=np.float64)
    
    
    correctTheta = np.zeros(360, dtype=float)
    correctDeflection = np.zeros(360, dtype=float)
    calcTheta = np.zeros(360, dtype=float)
    calcDeflection = np.zeros(360, dtype=float)
    xis = np.zeros(360, dtype=float)
    yis = np.zeros(360, dtype=float)
    calcCondition = np.zeros(360, dtype=float)
    
    tic360 = time.time()
    # trajectory to image points
    img = np.zeros((imgh, imgw, 3), dtype=np.uint8)
    steps = range(360)
    for i in range(360):
    #    bladePeak = np.array([r_blade * cos(i / (180./pi)), \
    #                r_blade * sin(i / (180./pi)), 0, 1], dtype=np.float64)
    #    bladePeakGlobal = np.matmul(r44inv_blade, bladePeak.transpose())
        theta = 1.0 * i
        deflection = 5.0 * sin(20.* i / 57.295779)
        bladeImagePoint = bladeImgPointByThetaAndDeflection( \
            theta, deflection, r_blade, r44_tur_wld, cmat, dvec, r44_cam_wld)
        yi = int(bladeImagePoint[1] + 0.5)
        xi = int(bladeImagePoint[0] + 0.5)
        
        calcTheta[i], calcDeflection[i], calcCondition[i] = \
            bladeThetaAndDeflectionByImgPoint_fastTrial(
                bladeImagePoint, \
                r_blade, r44_tur_wld, cmat, dvec, r44)
        correctTheta[i] = i
        correctDeflection[i] = deflection
        xis[i] = bladeImagePoint[0]
        yis[i] = bladeImagePoint[1] 

        bladeImagePoint_proj = bladeImgPointByThetaAndDeflection(calcTheta, calcDeflection, r_blade, r44_tur_wld, cmat, dvec, r44_cam_wld)
        yi_proj = int(bladeImagePoint_proj[1] + 0.5)
        xi_proj = int(bladeImagePoint_proj[0] + 0.5)
        
        if yi >= 0 and yi < imgh and xi >= 0 and xi < imgw:
            img[yi, xi] = [255,128,128]
        if yi >= 0 and yi < imgh and xi >= 0 and xi < imgw:
            img[yi_proj, xi_proj] = [128,128,255]

    
    # timing
    toc360 = time.time()
    print('Computing time: %f.2 sec.\n' % (toc360 - tic360))
    
    # plot
    deflPlot = plt.figure(1)
    plt.plot(steps, calcDeflection)
    thetPlot = plt.figure(2)
    plt.plot(steps, calcTheta)
    imgPlot = plt.figure(3)
    plt.imshow(img, cmap='gray')
    condPlot = plt.figure(4)
    plt.plot(range(360), calcCondition)


def turbineCalib_plotPointsAndManyPointProjection(xis, rvec, tvec, cmat, dvec, r_blade):
    nMany = 360
    x4d_many = np.zeros((nMany, 4), dtype=np.float)
    x4d_many[:,3] = 1.0
    thetas_many = np.linspace(0, 2 * pi, nMany)
    x4d_many[:,0] = r_blade * np.cos(thetas_many)
    x4d_many[:,1] = r_blade * np.sin(thetas_many)
    xis_many_prj, tmp = cv.projectPoints(x4d_many[:,0:3].copy(),
        rvec, tvec, cmat, dvec)                   
    xis_many_prj = xis_many_prj.reshape(nMany, 2)
    plt.plot(xis[:,0], xis[:,1], xis_many_prj[:,0], xis_many_prj[:,1]); 
    plt.gca().invert_yaxis(); plt.gca().grid('on'); plt.gca().axis('equal') 

def test_turbine_calib():
    # blade extrinsic parameters and radius 
    turbineCenter = np.array([0, -50, 100], dtype=np.float64)
    r44_tur_wld = extrinsicR44ByCamposYawPitch2( \
        campos=turbineCenter, yaw=180., pitch=90 + 0.0 * 180 / pi)   
    r44_wld_tur = np.linalg.inv(r44_tur_wld)                                                        
    r_blade = 55.0
    
    # camera extrinsic 
    camPos = np.array([ -100, -400, 10], dtype = np.float64)
    camAim = turbineCenter
    r44_cam_wld = extrinsicR44ByCamposAndAim(camPos, camAim)
    r44_wld_cam = np.linalg.inv(r44_cam_wld)
    rvec_cam_wld, rvecjoc = cv.Rodrigues(r44_cam_wld[0:3,0:3])
    tvec_cam_wld = r44_cam_wld[0:3,3]
    
    # synthetic camera intrinsic
    imgh = 1080; imgw = 1920;
    fx = fy = 3000
    cx = (imgw - 1) / 2.0; cy = (imgh - 1) / 2.0
    cmat = np.eye(3, dtype=np.float64)
    cmat[0,0] = fx; cmat[1,1] = fy; cmat[0,2] = cx; cmat[1,2] = cy;
    dvec = np.zeros((1,8), dtype=np.float64)
    
    # synthetic blade motion
    nPoint = 20
    thetas = np.linspace(0, 360, nPoint)
    correctTheta = np.zeros(nPoint, dtype=float)
    correctDeflection = np.zeros(nPoint, dtype=float)
    calcTheta = np.zeros(nPoint, dtype=float)
    calcDeflection = np.zeros(nPoint, dtype=float)
    xis = np.zeros((nPoint, 2), dtype=float)
    calcCondition = np.zeros(nPoint, dtype=float)
    
    # trajectory to image points
    for i in range(nPoint):
    #    bladePeak = np.array([r_blade * cos(i / (180./pi)), \
    #                r_blade * sin(i / (180./pi)), 0, 1], dtype=np.float64)
    #    bladePeakGlobal = np.matmul(r44inv_blade, bladePeak.transpose())
        theta = thetas[i]
        deflection = 0.0
        # deflection = 5.0 * sin(20.* i / 57.295779)
        bladeImagePoint = bladeImgPointByThetaAndDeflection( \
            theta, deflection, r_blade, r44_tur_wld, cmat, dvec, r44_cam_wld)
        xis[i,0] = bladeImagePoint[0]
        xis[i,1] = bladeImagePoint[1]
        
        
    # find a non-tilt ellipse that best fits the image points
    # [ecx, ecy], where the ellipse form is:
    # a x** 2 + b y**2 + c x + d y = 1
    matxy = np.zeros((nPoint, 4), dtype=np.float)
    for i in range(nPoint):
        matxy[i,0] = xis[i,0] ** 2
        matxy[i,1] = xis[i,1] ** 2
        matxy[i,2] = xis[i,0]
        matxy[i,3] = xis[i,1] 
    ata = np.matmul(matxy.transpose(), matxy)
    atb = np.matmul(matxy.transpose(), np.ones((nPoint,1), dtype=np.float64))
    coefs = np.matmul(np.linalg.inv(ata), atb).reshape(4)
    ecx = -0.5 * coefs[2] / coefs[0]
    ecy = -0.5 * coefs[3] / coefs[1]
    
    
    # roughly estimate angle of each image point xthetas
    # assuming camera and turbine are about face to face. 
    # That is, camera z is close to negative turbine z
    # and camera x is close to negative turbine x
    xthetas = np.zeros((nPoint), dtype=np.float)
    for i in range(nPoint):
        dx = (xis[i,0] - ecx) * sqrt(abs(coefs[0]))
        dy = (xis[i,1] - ecy) * sqrt(abs(coefs[1]))
        xthetas[i] = atan2(dy, -dx)
        
    # estimate 3d points of each image point x3d[iPoint,0:3] 
    # assuming turbine center is origin of turbine coordinate system
    x4d = np.zeros((nPoint, 4), dtype=np.float)
    x4d[:,0] = r_blade * np.cos(xthetas)
    x4d[:,1] = r_blade * np.sin(xthetas)
    x4d[:,2] = 0.    
    x4d[:,3] = 1.

    # estimate an initial guess of extrinsic parameters
    turbineExtrinsic = cv.solvePnP(x4d[:,0:3].copy(), xis, cmat, dvec)
    rvec_cam_blade_calib = turbineExtrinsic[1];
    tvec_cam_blade_calib = turbineExtrinsic[2];
    r44_cam_blade_calib = r44FromRvecTvec(rvec_cam_blade_calib, tvec_cam_blade_calib)
    r44_blade_cam_calib = np.linalg.inv(r44_cam_blade_calib)

    # transform r44_cam_blade_calib to r44_cam_wld_calib 
    r44_cam_wld_calib = np.matmul(r44_cam_blade_calib, r44_blade_wld)
    r44_wld_cam_calib = np.linalg.inv(r44_cam_wld_calib)
    
    # debug
    xis_prj, tmp = cv.projectPoints(x4d[:,0:3].copy(), \
        rvec_cam_blade_calib, tvec_cam_blade_calib, cmat, dvec)                       
    xis_prj = xis_prj.reshape(nPoint, 2)
    plt.plot(xis[:,0], xis[:,1], xis_prj[:,0], xis_prj[:,1]); 
    plt.gca().invert_yaxis(); plt.gca().grid('on'); plt.gca.axis('equal')   
    blade_center_calib = \
        np.linalg.inv(r44FromRvecTvec(rvec_cam_blade_calib, \
                                  tvec_cam_blade_calib))[0:3,3]
    xi_blade, tmp = cv.projectPoints(blade_center_calib, \
       rvec_cam_blade_calib, tvec_cam_blade_calib, cmat, dvec)                       
         
    # many-point projection
    nMany = 360
    x4d_many = np.zeros((nMany, 4), dtype=np.float)
    x4d_many[:,3] = 1.0
    thetas_many = np.linspace(0, 2 * pi, nMany)
    x4d_many[:,0] = r_blade * np.cos(thetas_many)
    x4d_many[:,1] = r_blade * np.sin(thetas_many)
    xis_many_prj, tmp = cv.projectPoints(x4d_many[:,0:3].copy(),
        rvec_cam_blade_calib, tvec_cam_blade_calib, cmat, dvec)                   
    xis_many_prj = xis_many_prj.reshape(nMany, 2)
    plt.plot(xis[:,0], xis[:,1], xis_many_prj[:,0], xis_many_prj[:,1]); 
    plt.gca().invert_yaxis(); plt.gca().grid('on'); plt.gca().axis('equal') 

    # least squares
    xRvecTvec0 = np.zeros(6, dtype=np.float)
    xRvecTvec0[0:3] = rvec_cam_blade_calib.reshape(3)
    xRvecTvec0[3:6] = tvec_cam_blade_calib.reshape(3)
    ptsMany3d = x4d_many[:,0:3]
    # res = funMinDistBetweenPoints2dAndManyPoints2dByRvecTvec(xRvecTvec0, cmat, dvec, ptsMany3d, xis)
    # 
    ticTurbineCalib = time.time()
    res_lsq = least_squares(funMinDistBetweenPoints2dAndManyPoints2dByRvecTvec, \
            xRvecTvec0, verbose = 1, \
            args=(cmat, dvec, ptsMany3d, xis), \
            gtol = 1e-3, max_nfev = 1000)
    rvec_cam_blade_calib_opt = res_lsq.x[0:3].reshape(3, 1)
    tvec_cam_blade_calib_opt = res_lsq.x[3:6].reshape(3, 1)
    r44_cam_blade_calib_opt = r44FromRvecTvec(rvec_cam_blade_calib_opt, \
                                              tvec_cam_blade_calib_opt)
    r44_blade_cam_calib_opt = np.linalg.inv(r44_cam_blade_calib_opt)
    tocTurbineCalib = time.time()
    timeTurbineCalib = tocTurbineCalib - ticTurbineCalib
    print('Turbine calibration time: %.2f sec.\n' % (timeTurbineCalib))


    turbineCalib_plotPointsAndManyPointProjection(xis, \
        rvec_cam_blade_calib_opt, tvec_cam_blade_calib_opt, cmat, dvec, r_blade)



