import numpy as np
import cv2
from scipy import ndimage as ndi

SEAM_COLOR = np.array([0, 0, 255])    # seam visualization color (BGR)
DOWNSIZE_WIDTH = 500                      # resized image width if downsize is True
ENERGY_MASK_CONST = 100000.0              # large energy value for protective masking
MASK_THRESHOLD = 10                       # minimum pixel intensity for binary mask
USE_FORWARD_ENERGY = True                 # if True, use forward energy algorithm
SHOW_ENERGY_MAP = False
SHOW_ALL_SEAM = False
LIST_SEAMS = []

########################################
# UTILITY CODE
########################################

def visualize(im, boolmask=None, rotate=False):
    vis = im.astype(np.uint8)
    if boolmask is not None:
        vis[np.where(boolmask == False)] = SEAM_COLOR
    if rotate:
        vis = rotate_image(vis, False)
    cv2.imshow("visualization", vis)
    cv2.waitKey(1)

def resize(image, width):
    dim = None
    h, w = image.shape[:2]
    dim = (width, int(h * width / float(w)))
    return cv2.resize(image, dim)

def rotate_image(image, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(image, k)    

########################################
# ENERGY FUNCTIONS
########################################

def backward_energy(im):
    """
    Simple gradient magnitude energy map.
    """
    xgrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap')
    ygrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap')
    
    grad_mag = np.sqrt(np.sum(xgrad**2, axis=2) + np.sum(ygrad**2, axis=2))

    return grad_mag

def forward_energy(im):
    """
    Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
    by Rubinstein, Shamir, Avidan.

    Vectorized code adapted from
    https://github.com/axu2/improved-seam-carving.
    """
    h, w = im.shape[:2]
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))
    
    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)
    
    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU
    
    for i in range(1, h):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)
        
        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)
        
    return energy

########################################
# SEAM HELPER FUNCTIONS
######################################## 

def add_seam(im, seam_idx):
    """
    Add a vertical seam to a 3-channel color image at the indices provided 
    by averaging the pixels values to the left and right of the seam.

    Code adapted from https://github.com/vivianhylee/seam-carving.
    """
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1, 3))
    for row in range(h):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                p = np.average(im[row, col: col + 2, ch])
                output[row, col, ch] = im[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]
            else:
                p = np.average(im[row, col - 1: col + 1, ch])
                output[row, : col, ch] = im[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]

    return output

def add_seam_grayscale(im, seam_idx):
    """
    Add a vertical seam to a grayscale image at the indices provided 
    by averaging the pixels values to the left and right of the seam.
    """    
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1))
    for row in range(h):
        col = seam_idx[row]
        if col == 0:
            p = np.average(im[row, col: col + 2])
            output[row, col] = im[row, col]
            output[row, col + 1] = p
            output[row, col + 1:] = im[row, col:]
        else:
            p = np.average(im[row, col - 1: col + 1])
            output[row, : col] = im[row, : col]
            output[row, col] = p
            output[row, col + 1:] = im[row, col:]

    return output

def remove_seam(im, boolmask):
    h, w = im.shape[:2]
    boolmask3c = np.stack([boolmask] * 3, axis=2)
    return im[boolmask3c].reshape((h, w - 1, 3))

def remove_seam_grayscale(im, boolmask):
    h, w = im.shape[:2]
    return im[boolmask].reshape((h, w - 1))

def get_minimum_seam(im, mask=None, remove_mask=None, rot=False):
    """
    DP algorithm for finding the seam of minimum energy. Code adapted from 
    https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
    """
    h, w = im.shape[:2]
    energyfn = forward_energy if USE_FORWARD_ENERGY else backward_energy
    M = energyfn(im)
    if SHOW_ENERGY_MAP:
        if rot:
            vis = rotate_image(M, False)
        else:
            vis = M
        cv2.imshow("Energy map", vis.astype(np.uint8))
        cv2.waitKey(1)
    
    if mask is not None:
        M[np.where(mask > MASK_THRESHOLD)] = ENERGY_MASK_CONST

    # give removal mask priority over protective mask by using larger negative value
    if remove_mask is not None:
        M[np.where(remove_mask < MASK_THRESHOLD)] = -ENERGY_MASK_CONST * 100

    backtrack = np.zeros_like(M, dtype=np.int)

    # populate DP matrix
    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    # backtrack to find path
    seam_idx = []
    boolmask = np.ones((h, w), dtype=np.bool)
    j = np.argmin(M[-1])
    for i in range(h-1, -1, -1):
        boolmask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx.reverse()
    return np.array(seam_idx), boolmask

########################################
# MAIN ALGORITHM
######################################## 

def seams_removal(im, num_remove, mask=None, vis=False, rot=False):
    for _ in range(num_remove):
        seam_idx, boolmask = get_minimum_seam(im, mask, rot=rot)
        if SHOW_ALL_SEAM: 
            LIST_SEAMS.append([rot, seam_idx])
        if vis:
            visualize(im, boolmask, rotate=rot)
        im = remove_seam(im, boolmask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolmask)
    return im, mask


def seams_insertion(im, num_add, mask=None, vis=False, rot=False):
    seams_record = []
    temp_im = im.copy()
    temp_mask = mask.copy() if mask is not None else None

    for _ in range(num_add):
        seam_idx, boolmask = get_minimum_seam(temp_im, temp_mask, rot=rot)
        if SHOW_ALL_SEAM: 
            LIST_SEAMS.append([rot, seam_idx])
        if vis:
            visualize(temp_im, boolmask, rotate=rot)

        seams_record.append(seam_idx)
        temp_im = remove_seam(temp_im, boolmask)
        if temp_mask is not None:
            temp_mask = remove_seam_grayscale(temp_mask, boolmask)

    seams_record.reverse()

    for _ in range(num_add):
        seam = seams_record.pop()
        im = add_seam(im, seam)
        if vis:
            visualize(im, rotate=rot)
        if mask is not None:
            mask = add_seam_grayscale(mask, seam)

        # update the remaining seam indices
        for remaining_seam in seams_record:
            remaining_seam[np.where(remaining_seam >= seam)] += 2         

    return im, mask

########################################
# MAIN DRIVER FUNCTIONS
########################################

def seam_carve(im, dy, dx, mask=None, vis=False):
    im = im.astype(np.float64)
    h, w = im.shape[:2]
    assert h + dy > 0 and w + dx > 0 and dy <= h and dx <= w

    if mask is not None:
        mask = mask.astype(np.float64)

    output = im

    if dx < 0:
        output, mask = seams_removal(output, -dx, mask, vis)

    elif dx > 0:
        output, mask = seams_insertion(output, dx, mask, vis)

    if dy < 0:
        output = rotate_image(output, True)
        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask = seams_removal(output, -dy, mask, vis, rot=True)
        output = rotate_image(output, False)

    elif dy > 0:
        output = rotate_image(output, True)
        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask = seams_insertion(output, dy, mask, vis, rot=True)
        output = rotate_image(output, False)

    return output


def object_removal(im, rmask, mask=None, vis=False, horizontal_removal=False, keep_ratio=False):
    im = im.astype(np.float64)
    rmask = rmask.astype(np.float64)
    if mask is not None:
        mask = mask.astype(np.float64)
    output = im

    h, w = im.shape[:2]

    if horizontal_removal:
        output = rotate_image(output, True)
        rmask = rotate_image(rmask, True)
        if mask is not None:
            mask = rotate_image(mask, True)

    while len(np.where(rmask < MASK_THRESHOLD)[0]) > 0:
        seam_idx, boolmask = get_minimum_seam(output, mask, rmask, rot=horizontal_removal)
        if SHOW_ALL_SEAM: 
            LIST_SEAMS.append([horizontal_removal, seam_idx])
        if vis:
            visualize(output, boolmask, rotate=horizontal_removal)            
        output = remove_seam(output, boolmask)
        rmask = remove_seam_grayscale(rmask, boolmask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolmask)
    
    if keep_ratio:
        num_add = (h if horizontal_removal else w) - output.shape[1]
        output, mask = seams_insertion(output, num_add, mask, vis, rot=horizontal_removal)

    if horizontal_removal:
        output = rotate_image(output, False)

    return output        

def drawAllSeams(im):
    h,w,_ = im.shape
    num_ver = sum(1 for seam in LIST_SEAMS if seam[0]==False)
    pre_seam_ver = [[100000]*h]
    pre_seam_hor = [[100000]*(w-num_ver)]
    
    for seam in LIST_SEAMS:
        if seam[0]:
            j = w-1
            for idx, i in enumerate(seam[1]):
                col = np.asarray(pre_seam_hor)[:,idx]
                i += sum(1 for cell in col if cell<=i)
                im[i][j] = SEAM_COLOR
                j -= 1
            pre_seam_hor.append(seam[1])
        else:
            i = 0
            for idx, j  in enumerate(seam[1]):
                row = np.asarray(pre_seam_ver).T[idx,:]
                j += sum(1 for cell in row if cell<=j)
                im[i][j] = SEAM_COLOR
                i += 1
            pre_seam_ver.append(seam[1])

    return im

def run_seam_carving(im, dx=0, dy=0, mask=None, rmask=None, hremove=False, kratio=False, mode='Forward', vis=False, vismask=False, vismap=False, visall=False, downsize=True):

    global USE_FORWARD_ENERGY, SHOW_ENERGY_MAP, SHOW_ALL_SEAM, LIST_SEAMS
    if mode=='Forward':
        USE_FORWARD_ENERGY = True 
    else: 
        USE_FORWARD_ENERGY = False
    if vismap:
        energyfn = forward_energy if USE_FORWARD_ENERGY else backward_energy
        M = energyfn(im.astype(np.float64))
        cv2.imwrite(f'map_{energyfn.__name__}.jpg', M.astype(np.uint8))
        SHOW_ENERGY_MAP = True
    else:
        SHOW_ENERGY_MAP = False
    if visall:
        SHOW_ALL_SEAM = True
    else:
        SHOW_ALL_SEAM = False
    LIST_SEAMS = []

    # downsize image for faster processing
    h, w = im.shape[:2]
    if mask is not None:
        mask = cv2.resize(mask, (w,h))
        if vismask:
            cv2.imshow('Protect Mask', mask)
            cv2.waitKey(1)
    if rmask is not None:
        rmask = cv2.resize(rmask, (w,h))
        if vismask:
            cv2.imshow('Remove Mask', rmask)
            cv2.waitKey(1)
    
    if downsize and w > DOWNSIZE_WIDTH:
        im = resize(im, width=DOWNSIZE_WIDTH)
        if mask is not None:
            mask = resize(mask, width=DOWNSIZE_WIDTH)
        if rmask is not None:
            rmask = resize(rmask, width=DOWNSIZE_WIDTH)

    output = im_all = None

    # object removal mode
    if rmask is not None:
        output = object_removal(im, rmask, mask, vis, hremove, kratio)

    # image resize mode
    if dx!=0 or dy!=0:
        if output is not None:
            output = seam_carve(output, dy, dx, mask, vis)
        else:
            output = seam_carve(im, dy, dx, mask, vis)
    cv2.destroyAllWindows()

    if SHOW_ALL_SEAM:
        im_all = drawAllSeams(im.copy())

    return output, im_all

