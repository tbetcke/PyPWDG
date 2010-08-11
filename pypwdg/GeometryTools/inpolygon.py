def pnpoly(xlist,ylist,xpoly,ypoly):
    """Test whether points are inside a polygon

    Input arguments:
    xlist - List of x-coordinates of test points
    ylist - List of y-coordinates of test points
    xpoly - List of x-coordinates of polygon
    ypoly - List of y-coordinates of polygon
    
    Output arguments:
    inpoly - Boolean array defined by inpoly[i]=1 if (xlist[i],ylist[i]) is inside
    the polygon defined by (xpoly,ypoly)
    """
    def point_inside_polygon(x,y,poly):
        """Test whether point x,y is in polygon defined by
           poly=[(x1,y1),(x2,y2),...]. Code adapted from
           http://www.ariel.com.au/a/python-point-int-poly.html
        """

        n = len(poly)
        inside =False

        p1x,p1y = poly[0]
        for i in range(n+1):
            p2x,p2y = poly[i % n]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x,p1y = p2x,p2y

        return inside

    plist=zip(xpoly,ypoly)
    poly_test=lambda x,y: point_inside_polygon(x,y,plist)
    return map(poly_test,xlist,ylist)


    
    
