__author__ = 'Joe'
import numpy as np
import sys
import cv2
import sharedmem as shmem
import numexpr as ne
from concurrent.futures import ThreadPoolExecutor


def get_log_kernel(siz, std):
    x = y = np.linspace(-siz, siz, 2*siz+1)
    x, y = np.meshgrid(x, y)
    arg = -(x**2 + y**2) / (2*std**2)
    h = np.exp(arg)
    h[h < sys.float_info.epsilon * h.max()] = 0
    h = h/h.sum() if h.sum() != 0 else h
    h1 = h*(x**2 + y**2 - 2*std**2) / (std**4)
    return h1 - h1.mean()


def get_edges(img, x):
    edges = list(range(0, img.shape[0], img.shape[0]//x))
    if img.shape[0] % x == 0:
        edges.append(img.shape[0])
    else:
        edges[-1] = img.shape[0]
    return edges


def fstack_mp_new(img, fmap):
    img_stacked = shmem.empty(img.shape[0:2], dtype='uint16')
    indexl = shmem.empty(img.shape[0:2], dtype='bool')

    edges = get_edges(img, 16)
    # This implementation is faster than breaking each image plane up for parallel processing
    def do_work(x):

        if x!=img.shape[2]-1:

            def mt_assignment(input, y):
                return input[index[edges[y]:edges[y+1],:]]
            index = ne.evaluate("fmap==x")
            img_stacked[index] = img[:, :, x][index]
            index = ne.evaluate("(fmap > x) & (fmap < x+1)")
            with ThreadPoolExecutor(max_workers=16) as pool:
                A = np.concatenate([(pool.submit(mt_assignment, fmap, y)).result() for y in range(16)], axis=0)
                B = np.concatenate([(pool.submit(mt_assignment, img[:, :, x+1], y)).result() for y in range(16)], axis=0)
                C = np.concatenate([(pool.submit(mt_assignment, img[:, :, x], y)).result() for y in range(16)], axis=0)
            print('A Shape is : ', A.shape)
            print('A content is: ', A)

            img_stacked[index] = ne.evaluate("(A-x) * B + (x+1-A) * C")
        else:
            last_ind = img.shape[2]-1
            indexl = ne.evaluate("fmap == last_ind")

    with shmem.MapReduce(np=img.shape[2]) as pool:
        pool.map(do_work, range(img.shape[2]))

    num_proc = shmem.cpu_count()
    edges = get_edges(img, num_proc)


    def mp_assignment(x):
        img_stacked[edges[x]:edges[x+1],:][indexl[edges[x]:edges[x+1],:]] = img[edges[x]:edges[x+1], :, -1]\
                                                                        [indexl[edges[x]:edges[x+1], :]]
    with shmem.MapReduce(np=num_proc) as pool:
        pool.map(mp_assignment, range(num_proc))

    return img_stacked


def fstack_mp(img, fmap):
    img_stacked = shmem.empty(img.shape[0:2], dtype='uint16')

    # This implementation is faster than breaking each image plane up for parallel processing
    def do_work(x):
        index = ne.evaluate("fmap==x")
        img_stacked[index] = img[:, :, x][index]
        index = ne.evaluate("(fmap > x) & (fmap < x+1)")
        A = fmap[index]
        B = img[:, :, x+1][index]
        C = img[:, :, x][index]
        img_stacked[index] = ne.evaluate("(A-x) * B + (x+1-A) * C")

    with shmem.MapReduce(np=img.shape[2]-1) as pool:
        pool.map(do_work, range(img.shape[2]-1))

    last_ind = img.shape[2]-1
    index = ne.evaluate("fmap == last_ind")
    num_proc = shmem.cpu_count()
    edges = get_edges(img, num_proc)

    def mp_assignment(x):
        img_stacked[edges[x]:edges[x+1],:][index[edges[x]:edges[x+1],:]] = img[edges[x]:edges[x+1], :, -1]\
                                                                        [index[edges[x]:edges[x+1], :]]
    with shmem.MapReduce(np=num_proc) as pool:
        pool.map(mp_assignment, range(num_proc))

    return img_stacked

def get_fmap(img):
    num_proc = shmem.cpu_count()
    log_kernel = get_log_kernel(11, 2)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))

    def mp_imgproc(x):
        bound_in = (edges[x]-(edges[x] > 0)*50, edges[x+1] + (edges[x+1] < img.shape[0]) *50 )
        bound_out = (50 if edges[x] > 0 else 0, None if edges[x+1] == img.shape[0] else -50)
        part_img = cv2.filter2D(img[bound_in[0]:bound_in[1], :, ii].astype('single'), -1, log_kernel)
        part_img = cv2.dilate(part_img, se)
        img_filtered[edges[x]:edges[x+1], :] = part_img[bound_out[0]:bound_out[1], :]

    def mp_gaussblur(x):
        bound_in = (edges[x]-(edges[x] > 0)*50, edges[x+1] + (edges[x+1] < img.shape[0]) *50 )
        bound_out = (50 if edges[x] > 0 else 0, None if edges[x+1] == img.shape[0] else -50)
        part_img = cv2.GaussianBlur(fmap[bound_in[0]:bound_in[1], :], (31, 31), 6)
        fmap[edges[x]:edges[x+1], :] = part_img[bound_out[0]:bound_out[1], :]

    log_response = shmem.empty(img.shape[0:2], dtype='single')
    fmap = shmem.empty(img.shape[0:2], dtype='single')
    edges = get_edges(img, num_proc)

    def mp_assignment_1(x):
        log_response[edges[x]:edges[x+1],:] = img_filtered[edges[x]:edges[x+1],:]

    def mp_assignment_2(x):
        fmap[index[edges[x]:edges[x+1],:]] = ii

    for ii in range(img.shape[2]):
        img_filtered = shmem.empty((img.shape[0], img.shape[1]), dtype='single')
        with shmem.MapReduce(np=num_proc) as pool:
            pool.map(mp_imgproc, range(num_proc))

        index = ne.evaluate("img_filtered > log_response")

        with shmem.MapReduce(np=num_proc) as pool:
            pool.map(mp_assignment_1, range(num_proc))
        # log_response[index] = img_filtered[index]
        with shmem.MapReduce(np=num_proc) as pool:
            pool.map(mp_assignment_2, range(num_proc))

    with shmem.MapReduce(np=num_proc) as pool:
        pool.map(mp_gaussblur, range(num_proc))
    return fmap
