def rotate_shape(shape, times=1):
    """
    Rotates the shape 90 degrees clockwise 'times' number of times.
    """
    for _ in range(times):
        shape = [ (y, -x) for x, y in shape ]
    # Normalize the shape to start at (0,0)
    min_x = min(x for x, y in shape)
    min_y = min(y for x, y in shape)
    shape = [ (x - min_x, y - min_y) for x, y in shape ]
    return shape

def mirror_shape(shape):
    """
    Mirrors the shape horizontally.
    """
    mirrored = [ (x, -y) for x, y in shape ]
    # Normalize
    min_x = min(x for x, y in mirrored)
    min_y = min(y for x, y in mirrored)
    mirrored = [ (x - min_x, y - min_y) for x, y in mirrored ]
    return mirrored
