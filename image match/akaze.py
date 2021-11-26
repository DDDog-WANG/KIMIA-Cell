import numpy as np
import cv2

def AKAZEexp(load,img1,img2):
    akaze = cv2.AKAZE_create()
    img1 = cv2.imread(load+img1)
    img2 = cv2.imread(load+img2)
    kp1, des1 = akaze.detectAndCompute(img1,None) 
    kp2, des2 = akaze.detectAndCompute(img2,None)

    #计算matches的平均距离
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
    matches = bf.match(des1,des2)
    dist_matches = [m.distance for m in matches]
    AKAZE_match_matchpoints=len(dist_matches)
    AKAZE_match_Averagedistance=np.mean(dist_matches)
    
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
        AKAZE_KnnMatch_matchpoints=np.count_nonzero(matchesMask)
        AKAZE_KnnMatch_Averagedistance=np.mean(dist_ransac)
        
    else:
        #计算KNN_RANSAC之后平均距离
        AKAZE_KnnMatch_matchpoints=0
        AKAZE_KnnMatch_Averagedistance=10000
    return AKAZE_match_matchpoints,AKAZE_match_Averagedistance,AKAZE_KnnMatch_matchpoints,AKAZE_KnnMatch_Averagedistance
