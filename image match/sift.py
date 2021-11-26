import numpy as np
import cv2

def SIFTexp(load,img1,img2):
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = cv2.imread(load+img1)
    img2 = cv2.imread(load+img2)
    kp1, des1 = sift.detectAndCompute(img1,None) 
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    #计算matches的平均距离
    bf = cv2.BFMatcher(crossCheck=False)
    matches = bf.match(des1,des2)
    dist_matches = [m.distance for m in matches]
    SIFT_match_matchpoints=len(dist_matches)
    SIFT_match_Averagedistance=np.mean(dist_matches)
    
    knnmatches = bf.knnMatch(des1,des2,k=2)    
    good = []
    for m,n in knnmatches:
        if m.distance < 0.8*n.distance:
            good.append(m)   
    if len(good) > 1:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 7.0)
        matchesMask = mask.ravel().tolist() 

        #计算KNN_RANSAC之后平均距离
        dist = [m.distance for m in good]
        dist_ransac=[]
        for m in range(len(good)):
            dist_ransac.append(dist[m]*matchesMask[m])
        SIFT_KnnMatch_matchpoints=np.count_nonzero(matchesMask)
        SIFT_KnnMatch_Averagedistance=np.mean(dist_ransac)
        
    else:
        #计算KNN_RANSAC之后平均距离
        SIFT_KnnMatch_matchpoints=0
        SIFT_KnnMatch_Averagedistance=10000
    return SIFT_match_matchpoints,SIFT_match_Averagedistance,SIFT_KnnMatch_matchpoints,SIFT_KnnMatch_Averagedistance    