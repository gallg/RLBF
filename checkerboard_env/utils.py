from PIL import Image
from PIL.ImageOps import invert

def load_checkerboard(img, cross):
    
    board = Image.open(img)
    cross = Image.open(cross)

    if board.mode == 'RGBA':
        r,g,b,a = board.split()
        rgb_image = Image.merge('RGB', (r,g,b))
        rgb_inverse = invert(rgb_image)

        r2,g2,b2 = rgb_inverse.split()
        inverse = Image.merge('RGBA', (r2,g2,b2,a))

    else:
        inverse = invert(board)
    
    return board, inverse, cross
