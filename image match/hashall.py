import cv2
import numpy as np

#均值哈希算法
def aHash(imgname):
    img=cv2.resize(imgname,(8,8))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    np_mean = np.mean(gray)                           # 求numpy.ndarray平均值
    ahash_01 = (gray>np_mean)+0                       # 大于平均值=1，否则=0
    ahash_list = ahash_01.reshape(1,-1)[0].tolist()   # 展平->转成列表
    ahash_str = ''.join([str(x) for x in ahash_list])
    return ahash_str

def pHash(imgname):
    img = cv2.resize(imgname, (32, 32))    # 默认interpolation=cv2.INTER_CUBIC
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_roi = dct[0:8, 0:8]            # opencv实现的掩码操作
    avreage = np.mean(dct_roi)
    phash_01 = (dct_roi>avreage)+0
    phash_list = phash_01.reshape(1,-1)[0].tolist()
    phash_str = ''.join([str(x) for x in phash_list])
    return phash_str

def dHash(imgname):
    img=cv2.resize(imgname,(9,8))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #每行前一个像素大于后一个像素为1，相反为0，生成哈希
    hash_str0 = []
    for i in range(8):
        hash_str0.append(gray[:, i] > gray[:, i + 1])
    hash_str1 = np.array(hash_str0)+0
    hash_str2 = hash_str1.T
    hash_str3 = hash_str2.reshape(1,-1)[0].tolist()
    dhash_str = ''.join([str(x) for x in hash_str3])
    return dhash_str

def hammingDist(s1, s2):
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])

def aHashexp(load,img1,img2):
    return 1 - hammingDist(aHash(load+img1),aHash(load+img2))* 1. / (32 * 32 / 4)

def pHashexp(load,img1,img2):
    return 1 - hammingDist(pHash(load+img1),pHash(load+img2))* 1. / (32 * 32 / 4)

def dHashexp(load,img1,img2):
    return 1 - hammingDist(dHash(load+img1),dHash(load+img2))* 1. / (32 * 32 / 4)