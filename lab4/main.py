import cv2
import os
from tqdm import tqdm
import time

test_original = cv2.imread("Altered-custom/1__M_Left_index_finger_szum.BMP")


# algorytm, algorytm matchingu, flann trees, knn N, distance threshold
# -> correct?, match %, num match points, num keypoints

# SIFT      flann               10           2      0.1
# -> 


dataDir = "Real_subset"


def preprocess(image):
    # preprocessing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.equalizeHist(image)
    # image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def features_extraction(image):
    # feature extraction - sift, surf, fast, brief ...
    return feature_algo.detectAndCompute(image, None)


test_preprocessed = preprocess(test_original)
# cv2.imshow("Original", cv2.resize(test_preprocessed, None, fx=1, fy=1))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

feature_algo = cv2.SIFT_create()
keypoints_1, descriptors_1 = features_extraction(test_preprocessed)

total = 0
correct_matches = 0
avg_match = 0
avg_matched_points = 0
avg_keypoints = 0
t_start = time.time()

for file in tqdm(os.listdir("Real_subset")):
    total += 1
    # print('\n\n------')
    # print('file:', file)

    fingerprint_database_image = cv2.imread("./Real_subset/" + file)

    keypoints_2, descriptors_2 = features_extraction(preprocess(fingerprint_database_image))

    # matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=1),
    #                                 dict()).knnMatch(descriptors_1, descriptors_2, k=2)

    matches = cv2.BFMatcher().knnMatch(descriptors_1, descriptors_2, k=2)

    match_points = []

    for p, q in matches:
        if p.distance < 0.2 * q.distance:
            match_points.append(p)

    keypoints = 0
    if len(keypoints_1) <= len(keypoints_2):
        keypoints = len(keypoints_1)
    else:
        keypoints = len(keypoints_2)
    avg_keypoints += keypoints
    if (len(match_points) / keypoints) > 0:
        # print("% match: ", len(match_points) / keypoints * 100)
        # print("Fingerprint ID: " + str(file))
        result = cv2.drawMatches(test_original, keypoints_1, fingerprint_database_image,
                                 keypoints_2, match_points, None)
        result = cv2.resize(result, None, fx=2.5, fy=2.5)
        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(str(len(match_points)) + " " + str(keypoints))

        avg_match += len(match_points) / keypoints * 100
        avg_matched_points += len(match_points)
        correct_matches += 1

elapsed_time = time.time() - t_start


print("Elapsed time: ", elapsed_time * 1000, "ms")
print("Total: ", total)
print("Matches: ", correct_matches)
print("Avg match: ", avg_match / correct_matches, "%")
print("Avg matched points: ", avg_matched_points / correct_matches)
print("Avg keypoints: ", avg_keypoints / total)
