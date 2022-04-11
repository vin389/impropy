"""
    This function calculates the minimum distance between each given point and 
    a given polygonal chain. 
    
    Paramters:
        Xs: TYPE np.array((n, 2), dtype = float), where n is number of points
          the coordinates of all given points
        Xpc: TYPE np.array(m, 2), dtype = float), where m is the number of 
          points that define the polygonal chain
    Returns:
        dists: TYPE np.array(n, dtype = float), where n is number of points
          the minimum distance between each given point and the polygonal chain
"""
def distancePointsToPolygonalChain(Xs, Xpc):
    return 0


"""
    This function calculates the distance between a given point and a given 
    line, a parameter alpha, and a validation flag. 
    
    Parameters:
        xn : TYPE float 
          the x coordinate of point n, the given point
        yn : TYPE float 
          the y coordinate of point 0, the given point
        x0 : TYPE float 
          the x coordinate of point 0, one of the end point of the line section
        y0 : TYPE float 
          the y coordinate of point 0, one of the end point of the line section
        x1 : TYPE float 
          the x coordinate of point 1, the other point of the line section
        y1 : TYPE float 
          the y coordinate of point 0, the other point of the line section
          
    Returns:
        d : TYPE float
          the distance between the given point and the line defined by 
          (x0, y0) and (x1, y1)
        alpha : TYPE float
          a parameter that describes where the point along the line which is 
          most close to (xn, yn). If alpha is 0, the projection point is at 
          (x0, y0). If alpha is 1, the projection point is at (x1, y1). If 
          alpha < 0, the projection point is out of the line section of 
          (x0, y0) and (x1, y1), and is at the side of (x0, y0). See the 
          figure of the documentation of this function. 
        valid : TYPE logic
          If (x0, y0) and (x1, y1) are two distinct points, the returned valid
          would be True. Otherwise, valid would be False. 
          
    See https://docs.google.com/presentation/d/1rAMlIODW3EWhpcU8_psVJz5sTh8li2HhzFqwkIGKVC4/ 
        for more details
      
"""
def distancePointTwoLineSection(xn, yn, x0, y0, x1, y1):
    dx = x0 - x1
    dy = y0 - y1
    # check if two points are exactly the same point 
    if (dx == 0.0 and dy == 0.0):
        return 0, 0, False 
    # calculate alpha 
    alpha = (dx * (x0 - xn) + dy * (y0 - yn)) / (dx ** 2 + dy ** 2)
    # calculate distance
    d2 = (xn - (1 - alpha) * x0 + alpha * x1) ** 2 + \
         (yn - (1 - alpha) * y0 + alpha * y1) ** 2 
    d = d2 ** 0.5
    return d, alpha, True

def distancePointTwoLineSectionV2(x0,y0,x1,y1,xn,yn):
    dx = x1 - x0
    dy = y1 - y0
    # check if two points are exactly the same point 
    if (dx == 0.0 and dy == 0.0):
        return 0, 0, False 
    # calculate line 0
    a0 = dy 
    b0 = -dx
    c0 = -a0 * x0 - b0 * y0
    e0 = a0 * xn + b0 * yn + c0
    d02 = e0 * e0 / (a0 ** 2 + b0 ** 2)
    d0 = d02 ** 0.5
    # calculate line 1
    a1 = dx 
    b1 = dy
    xk = 0.5 * (x0 + x1)
    yk = 0.5 * (y0 + y1)
    c1 = -a1 * xk - b1 * yk
    e1 = a1 * xn + b1 * yn + c1
    d12 = e1 * e1 / (a1 ** 2 + b1 ** 2)
    d1 = d12 ** 0.5
    # 
    return d0, d1, True


"""
    This function calculates .... 
    
    Parameters:
        x0: TYPE float
          the x coordinate of the first point 
"""

def distancePointToPoint(x0, y0, x1, y1):
    dx = (x0 - x1)
    dy = (y0 - y1)
    return (dx ** 2 + dy ** 2) ** 0.5
    
    
    
