import numpy as np
import cv2 
import time

train_data = np.load('MUSHR_320x240_shuffled_0.npy',allow_pickle=True)
# output = np.load('MUSHR_320x240_5_outputs.npy',allow_pickle=True)

X = np.array([i[0] for i in train_data])
print(X.shape)
Y = np.array([i[1] for i in train_data])
print(Y.shape)
cv2.namedWindow('visualize',cv2.WINDOW_NORMAL)
cv2.resizeWindow('visualize',800,600)
for i in range(0,train_data.shape[0],10):
	added_image = cv2.addWeighted(X[i],0.8,Y[i],0.2,0)
	cv2.imshow('visualize',added_image)
	cv2.waitKey(10)
	time.sleep(0.5)

# cv2.destroyAllWindows()