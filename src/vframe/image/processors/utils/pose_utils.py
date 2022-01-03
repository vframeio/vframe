import numpy as np
import cv2 as cv

from vframe.settings.app_cfg import LOG
from vframe.models.geometry import BBox

N_POINTS = 18

# COCO Output Format
KEYPOINTS_MAPPING = ['nose', 'neck', 'shoulder_right', 'elbow_right', 'wrist_right', 'shoulder_left', 
  'elbow_left', 'wrist_left', 'hip_right', 'knee_right', 'ankle_right', 'hip_left', 'knee_left', 
  'ankle_left', 'eye_right', 'eye_left', 'ear_right', 'ear_left']

#KEYPOINTS_MAPPING_DISPLAY = ['Nose', 'Neck', 'Shoulder Right', 'Elbow Right', 'Wrist Right', 'Shoulder Left', 
#  'Elbow Left', 'Wrist Left', 'Hip Right', 'Knee Right', 'Ankle Right', 'Hip Left', 'Knee Left', 
#  'Ankle Left', 'Eye Right', 'Eye Left', 'Ear Right', 'Ear Left']

KEYPOINTS_MAPPING_ABBREV = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
  'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 
  'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']


POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16]]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
MAP_IDX = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

COLORS = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]




def probmap_to_keypoints(im_probmap, threshold=0.1, k=3):
  """Converts probability grayscale image (map) to keypoints
  """
  im_smooth = cv.GaussianBlur(im_probmap, (k,k), 0, 0)
  im_mask = np.uint8(im_smooth > threshold)
  keypoints = []

  # find blobs
  contours, _ = cv.findContours(im_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

  # find blob maxima
  for cnt in contours:
      im_blob_mask = np.zeros(im_mask.shape)
      im_blob_mask = cv.fillConvexPoly(im_blob_mask, cnt, 1)
      im_mask_probmap = im_smooth * im_blob_mask
      _, val_max, _, loc_max = cv.minMaxLoc(im_mask_probmap)
      keypoints.append(loc_max + (im_probmap[loc_max[1], loc_max[0]],))

  return keypoints



def mk_valid_pairs(output, dim, detected_keypoints):
  """Find valid connections between the different joints of a all persons present
  """  
  valid_pairs = []
  invalid_pairs = []
  n_interp_samples = 10  # 10
  paf_score_th = 0.1  # 0.1
  conf_th = 0.75  # 0.75

  # loop for every POSE_PAIR
  for k in range(len(MAP_IDX)):
    # A->B constitute a limb
    paf_a = output[0, MAP_IDX[k][0], :, :]
    paf_b = output[0, MAP_IDX[k][1], :, :]
    paf_a = cv.resize(paf_a, dim)
    paf_b = cv.resize(paf_b, dim)

    # Find the keypoints for the first and second limb
    cand_a = detected_keypoints[POSE_PAIRS[k][0]]
    cand_b = detected_keypoints[POSE_PAIRS[k][1]]
    n_a = len(cand_a)
    n_b = len(cand_b)

    """
    If keypoints for the joint-pair is detected
      check every joint in candA with every joint in candB
      Calculate the distance vector between the two joints
      Find the PAF values at a set of interpolated points between the joints
      Use the above formula to compute a score to mark the connection valid
    """

    if( n_a != 0 and n_b != 0):
      valid_pair = np.zeros((0,3))

      for i in range(n_a):
        
        max_j = -1
        score_max = -1
        found = 0

        for j in range(n_b):
          # Find d_ij
          d_ij = np.subtract(cand_b[j][:2], cand_a[i][:2])
          norm = np.linalg.norm(d_ij)

          if norm:
            d_ij = d_ij / norm
          else:
            continue
          
          # Find p(u)
          interp_coord = list(zip(np.linspace(cand_a[i][0], cand_b[j][0], num=n_interp_samples), 
            np.linspace(cand_a[i][1], cand_b[j][1], num=n_interp_samples)))

          # Find L(p(u))
          paf_interp = []
          for k in range(len(interp_coord)):
            paf_interp.append([paf_a[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
              paf_b[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])

          # Find E
          paf_scores = np.dot(paf_interp, d_ij)
          avg_paf_score_avg = sum(paf_scores)/len(paf_scores)

          # check if connection is valid
          # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
          if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th:
            if avg_paf_score_avg > score_max:
              max_j = j
              score_max = avg_paf_score_avg
              found = 1

          # Append the connection to the list
          if found:
            valid_pair = np.append(valid_pair, [[cand_a[i][3], cand_b[max_j][3], score_max]], axis=0)

      # Append the detected connections to the global list
      valid_pairs.append(valid_pair)

    else:
      # no keypoints are detected
      invalid_pairs.append(k)
      valid_pairs.append([])

  return valid_pairs, invalid_pairs


def mk_personwise_keypoints(valid_pairs, invalid_pairs, keypoints_list):
  """Creates a list of keypoints belonging to each person
    For each detected valid pair, it assigns the joint(s) to a person
    the last number in each row is the overall score
  """
  personwise_keypoints = -1 * np.ones((0, 19))

  for k in range(len(MAP_IDX)):
    if k not in invalid_pairs:
      parts_a = valid_pairs[k][:,0]
      parts_b = valid_pairs[k][:,1]
      index_a, index_b = np.array(POSE_PAIRS[k])

      for i in range(len(valid_pairs[k])):
        found = 0
        person_idx = -1
        for j in range(len(personwise_keypoints)):
          if personwise_keypoints[j][index_a] == parts_a[i]:
            person_idx = j
            found = 1
            break

        if found:
          personwise_keypoints[person_idx][index_b] = parts_b[i]
          personwise_keypoints[person_idx][-1] += keypoints_list[parts_b[i].astype(int), 2] + valid_pairs[k][i][2]

        # if find no partA in the subset, create a new subset
        elif not found and k < 17:
          row = -1 * np.ones(19)
          row[index_a] = parts_a[i]
          row[index_b] = parts_b[i]
          # add the keypoint_scores for the two keypoints and the paf_score
          row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
          personwise_keypoints = np.vstack([personwise_keypoints, row])

  return personwise_keypoints


def mk_detected_keypoints(output, dim, threshold=0.1, k=3):

  detected_keypoints = []
  keypoints_list = np.zeros((0,3))
  keypoint_id = 0

  for part in range(N_POINTS):
    im_probmap = output[0,part,:,:]
    im_probmap = cv.resize(im_probmap, dim)
    keypoints = probmap_to_keypoints(im_probmap, threshold=threshold, k=k)
    keypoints_with_id = []
    for i in range(len(keypoints)):
      keypoints_with_id.append(keypoints[i] + (keypoint_id, KEYPOINTS_MAPPING[part]))
      keypoints_list = np.vstack([keypoints_list, keypoints[i]])
      keypoint_id += 1

    detected_keypoints.append(keypoints_with_id)

  return detected_keypoints, keypoints_list

  
def keypoints_to_persons(detected_keypoints, personwise_keypoints):
  """Groups keypoints into list of person-pose dicts
  """
  persons = []
  n_persons = len(personwise_keypoints)

  # rebuild as dict of each person's keypoints
  detected_keypoints_lkup = {}
  for part in detected_keypoints:
    for kpts in part:
      detected_keypoints_lkup[kpts[3]] = (kpts[0], kpts[1], kpts[4], kpts[2])

  persons_kpt_idxs = []
  for n in range(n_persons):
    kpts = []
    for i in range(17):
      kpts += personwise_keypoints[n][np.array(POSE_PAIRS[i])].astype(int).tolist()
    kpts = list(set(kpts))
    if -1 in kpts:
      kpts.pop(-1)
    persons_kpt_idxs.append(kpts)

  # create list of person dicts with label and keypoints
  for i in range(n_persons):
    pose = {detected_keypoints_lkup[x][2]: detected_keypoints_lkup[x][:2] for x in persons_kpt_idxs[i] if x != -1}
    persons.append(pose)

  return persons


def pose_to_bbox(pose, dim):
  """Estimates a face bbox given a partial pose
  :param pose: a dict with label keys and keypoint values ( eg {"nose": (100,100)} )
  :returns BBox
  """

  # get points
  nose = pose.get('nose')
  shoulder_left = pose.get('shoulder_left')
  shoulder_right = pose.get('shoulder_right')
  eye_left = pose.get('eye_left')
  eye_right = pose.get('eye_right')
  ear_left = pose.get('ear_left')
  ear_right = pose.get('ear_right')
  neck = pose.get('neck')

  # assume head exists if these keypoints exist
  head_exists = nose and neck and (shoulder_right or shoulder_left) and (eye_left or eye_right)  # strict
  #head_exists = nose and neck and (shoulder_right or shoulder_left)  # general

  if not head_exists:
    return None

  # x1 best to worst estimates
  if ear_right:
    x1 = ear_right[0]
  elif eye_right and shoulder_right:
    # use right shoulder distance to estimated ear position
    x1 = min(eye_right[0], eye_right[0] - (0.5 * max(0, abs(shoulder_right[0] - eye_right[0]))))
  elif eye_right:
    # double the distance between nose and eye
    x1 = min(eye_right[0], eye_right[0] - 2 * abs(nose[0] - eye_right[0]))
  elif eye_left:
    # mirror left eye x to get right eye x
    x1 = min(nose[0], nose[0] - 2 * abs(nose[0] - eye_left[0]))
  else:
    # use nose to neck distance to estimate x1
    x1 = min(nose[0], nose[0] - (0.35 * abs(nose[0] - neck[0])))
  
  # x2 best to worst estimates
  if ear_left:
    x2 = ear_left[0]
  elif eye_left and shoulder_left:
    # use right shoulder distance to estimated ear position
    x2 = max(eye_left[0], eye_left[0] + (0.5 * max(0, abs(shoulder_left[0] - eye_left[0]))))
  elif eye_left:
    # double the distance between nose and eye
    x2 = max(eye_left[0], eye_left[0] + 2 * abs(nose[0] - eye_left[0]))
  elif eye_right:
    # mirror right eye x to right
    x2 = max(nose[0], nose[0] + 2 * abs(nose[0] - eye_right[0]))
  else:
    # use nose to neck distance to estimate x2
    x2 = max(nose[0], nose[0] + (0.35 * abs(nose[0] - neck[0])))

  # y1, y2 from neck and head width
  hh = 1.15 * abs(nose[1] - neck[1])
  y1 = min(max(0, nose[1] - hh // 2), dim[0])
  y2 = min(max(0, nose[1] + hh // 2), dim[1])
  # expand to facial keypoints for strongly rotated faces
  if ear_left:
    y1 = min(y1, ear_left[1])
    y2 = max(y2, ear_left[1])
  if ear_right:
    y1 = min(y1, ear_right[1])
    y2 = max(y2, ear_right[1])

  # make bbox
  xyxy = list(map(int, [x1, y1, x2, y2]))
  bbox = BBox(*xyxy, *dim)
  if bbox.width < 12 or bbox.height < 12:
    return None
  return bbox


def persons_to_bboxes(persons, dim):
  """Estimates face BBox from pose keypoints
  :param persons: list
  :param dim: tuple of width x height
  :returns list of BBox
  """
  bboxes = []
  for i, pose in enumerate(persons):
    bbox = pose_to_bbox(pose, dim)
    if bbox:
      bboxes.append(bbox)
  return bboxes

