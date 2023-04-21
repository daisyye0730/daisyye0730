import cv2
import numpy as np
from shapely.geometry import Polygon
import math
from collections import Counter


# Constants 
CAMPUS = "Campus.pgm"
LABEL = "Labeled.pgm"
TABLE = "Table.txt"
MED_DIAGONAL = 70
LG_DIAGONAL = 110
NARROW = 0.33
MED_WIDTH = 0.66
CHECK_RECT_RATIO = 0.3 # this is to check the geometry of the building and see if it is rectangular 
ORIENTATION = 0.3


'''Processes the Table.txt file so it returns a dictionary with key being number, and value being the name'''
def processTableTxt():
    d = {}
    d2 = {}
    with open(TABLE) as f:
        for line in f:
            res = line.strip().split(' ')
            d[int(res[0])] = res[1]
            d2[res[1]] = int(res[0])
    return (d, d2)


'''Step 1: Raw Data'''
def printRawData(table, campus_img, label_img):
    # calculate total area in pixels by counting the number of pixels of a certain building  
    area = {}
    for R in range (0, len(label_img)):
        for C in range (0, len(label_img[0])):
            pix = label_img[R][C]
            if pix.all() == 0:
                continue 
            if pix[0] in area: 
                area[pix[0]] += 1
            else: 
                area[pix[0]] = 1

    # calculate center of mass 
    com = {}
    # calculate Minimum Bounding Rectangle 
    mbr = {}
    # a dictionary of contours and their buildings
    b_contours = {}
    imgray = cv2.cvtColor(campus_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i in range (0, len(contours)):
        c = contours[i]
        # this is to ignore the case when there is a contour inside a bigger contour 
        # so when there is empty space in the building
        if hierarchy[0][i][-1] != -1:
            continue
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = round(M['m10']/M['m00'], 2)
            cy = round(M['m01']/M['m00'], 2)
        # use a random point of the contour 
        # and locate the point in the label image to find which building the center of mass belongs to
        x, y = c[0][0]
        com[label_img[y][x][0]] = (cx, cy)
        
        b_contours[label_img[y][x][0]] = c
        
        # calculate Minimum Bounding Rectangle by finding minimum area rectangle and then finding its coordinates 
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # the diagonal 
        dist = round(np.linalg.norm(box[0]-box[2]), 2)
        mbr[label_img[y][x][0]] = (box, dist)
    
    # calculate MBR intersections of one building 
    mbr_intersect = {}
    for b1, valA in mbr.items():
        mbr_intersect[b1] = []
        boxA = valA[0]
        for b2, valB in mbr.items():
            if b2 == b1: 
                continue 
            boxB = valB[0]
            # create two polygons of the two boxes 
            polygon = Polygon([(boxA[0][0], boxA[0][1]), (boxA[1][0], boxA[1][1]), (boxA[2][0], boxA[2][1]), (boxA[3][0], boxA[3][1])])
            other_polygon = Polygon([(boxB[0][0], boxB[0][1]), (boxB[1][0], boxB[1][1]), (boxB[2][0], boxB[2][1]), (boxB[3][0], boxB[3][1])])
            # check the intersection area 
            intersection = polygon.intersection(other_polygon)
            if intersection.area > 0: 
                mbr_intersect[b1].append(table[b2])
                
    # print overall result 
    for key, val in table.items():
        print("Building #%d: %s" %(key, val))
        print("\tCenter of Mass: (%s, %s)" %(str(com[key][0]),str(com[key][1])))
        print("\tArea: ", area[key])
        print("\tMin Bounding Rect and diagonal: (%s,%s) to (%s,%s) with diagonal %s" \
           %(str(mbr[key][0][0][0]),str(mbr[key][0][0][1]),str(mbr[key][0][2][0]),str(mbr[key][0][2][1]), str(mbr[key][1])))
        if len(mbr_intersect[key]) == 0: 
            print("\tIntersection Buildings: None")
        else:
            print("\tIntersection Buildings: ", mbr_intersect[key])
    return (area, com, mbr, mbr_intersect, b_contours)
            

'''Step 2: "What"'''
def describeShape(table, area, com, mbr, mbr_intersect, description, contours):
    size(area, mbr, description)
    aspectRatio(mbr, description)
    geometry(table, area, description, contours, mbr)


'''Describe geometry: square / rectangular / I-shaped / C-shaped / L-shaped / asymmetric'''
def geometry(t, area, description, contours, mbr):
    # first categorize shapes that are obviously rectangular and squarish 
    for k, cnt in contours.items():
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio= float(w)/h
            if ratio>=0.9 and ratio<=1.1:
                description[k].append("square")
            else:
                description[k].append("rectangular")
    im = cv2.imread(CAMPUS)
    
    # check the rest of the building one by one 
    for k, cnt in contours.items():
        if len(description[k]) == 3:
            continue
        # blank contains only one building at a time 
        blank = np.zeros(im.shape[0:2])
        blank = cv2.fillPoly(blank, [cnt], 1)
        if processI_shape(blank, area[k], com[k]): 
            description[k].append("I-shaped")
            continue 
        elif processC_shape(blank, t[k], com[k], mbr[k]):
            description[k].append("C-shaped")
            continue 
        elif processL_shape(blank):
            description[k].append("L-shaped")
            continue 
        val, shape = check_rect(blank, mbr[k], area[k])
        if val and shape == 'square':
            description[k].append("square")
        elif val and shape == 'rectangle': 
                description[k].append("rectangular")
        else:
            description[k].append("asymmetrical")


'''Helper geometry function to see if a shape is rectangular or square'''
def check_rect(blank, mbr, area):
    # check how different the shape is from its bounding rectangular box 
    box = mbr[0]
    err = 0
    for i in range (box[0][0], box[1][0]):
        for j in range (box[0][1], box[2][1]):
            if blank[j][i] != 1:
                err += 1 
    if err/area > CHECK_RECT_RATIO:
        return (False, None)
    # check if it a square or rectangle 
    l = math.dist(box[0], box[1])
    w = math.dist(box[3], box[0])
    if abs((l-w)/min(l, w)) < 0.1:
        return (True, 'square')
    return (True, 'rectangle')


'''Helper function to test if a shape looks like an "I"'''
def processI_shape(b, area, com):
    v_symm = int(round(com[0]))
    h_symm = int(round(com[1]))
    new_b = b.copy()
    # try cutting it vertically to see if it is still symmetrical 
    # change half of the shape to black
    for i in range (0, len(b)):
        for j in range (0, len(b[0])):
            if b[i][j] == 0:
                continue 
            if j > v_symm:
                new_b[i][j] = 0
    # other_side is the side that is still white 
    other_side = cv2.subtract(b, new_b)
    err = 0
    # check if a white pixel on one side means a white pixel on the other side 
    # if they don't match, there is an error 
    for i in range (0, len(new_b)):
        for j in range (0, len(new_b[0])):
            this_pix = new_b[i][j]
            if this_pix == 0:
                continue
            if j+2*(v_symm-j) >= 275 or other_side[i][j+2*(v_symm-j)] == 0:
                err += 1
    # standardize the error given its area
    if err/area > 0.1:
        return False
    # try cutting it horizontally to see if it is symmetrical 
    # the procedure is similar to the procedure above 
    blank_v = new_b.copy()
    for i in range (0, len(b)):
        for j in range (0, len(b[0])):
            if b[i][j] == 0:
                continue 
            if i > h_symm:
                blank_v[i][j] = 0
    other_side = cv2.subtract(blank_v, b)
    err = 0
    for i in range (0, len(b)):
        for j in range (0, len(b[0])):
            this_pix = blank_v[i][j]
            if this_pix == 0:
                continue
            if i+2*(h_symm-i) >= 495 or other_side[i+2*(h_symm-i)][j] == 0:
                err += 1
    if err/area > 0.01:
        return False
    # check if the resulting shape is a narrow L or a wide L
    # the actual I shape should be a narrow L at this stage 
    image = blank_v.astype('uint8')
    cnt, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _,_,w,h = cv2.boundingRect(cnt[0])
    if w<h:
        return True


'''Helper function to see if a shape looks like a C'''
def processC_shape(b, k, com, mbr):
    # first check how different the symmetries from the center of mass looks from the symmetries from the MBR
    # the logic behind this is that the two different symmetries (one from com, one from mbr) should not vary by much if it is symmetrical regarding that axis 
    # this is to eliminate the buildings that are not horizontally or vertically symmetrical 
    box = mbr[0]
    l = math.dist(box[0], box[1])
    w = math.dist(box[3], box[0])
    v = box[0][0]+int(round(abs(box[0][0]-box[1][0])/2))
    ho = box[0][1]+int(round(abs(box[0][1]-box[2][1])/2))
    vertical_symm = int(round(com[0]))
    horizontal_symm = int(round(com[1]))
    if abs(ho - horizontal_symm)/w > 0.04 and abs(v-vertical_symm)/l > 0.04:
        # eliminate buildings that are not horizontally or vertically symmetrical 
        return False
    # rotate the image by 90 degree unless the C is lying flat 
    b_flipped = cv2.rotate(b, cv2.ROTATE_90_CLOCKWISE)
    # try cutting it horizontally to see if it can result in L shape 
    blank_h = b.copy()
    blank_v = b_flipped.copy()
    # saving only the half of the original shape
    # if the building is C-shaped, this will result in a shape that looks like L  
    for i in range (0, len(b)):
        for j in range (0, len(b[0])):
            if b[i][j] == 0:
                continue 
            if i > horizontal_symm:
                blank_h[i][j] = 0
    for i in range (0, len(b_flipped)):
        for j in range (0, len(b_flipped[0])):
            if b_flipped[i][j] == 0:
                continue 
            if i > vertical_symm:
                blank_v[i][j] = 0
    # check its symmetry against its diagonal axis 
    return processL_shape(blank_v) or processL_shape(blank_h)


'''Helper function that checks if a building is shaped like a L'''
def processL_shape(blank):
    image = blank.astype('uint8')
    cnt, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(cnt[0])
    # cropped is the cropped image of a building that only contains its upper half
    # the idea is that if you flip an L shape against the major axes, it will lead to hollow heart rectangle
    cropped = blank[y:y+h, x:x+w]
    cropped_180 = cv2.flip(cropped, 1)
    and_cropped = np.concatenate((cropped, cropped_180), axis=1)
    # however, the L shapes can be facing opposite directions, 
    # therefore, we also need to see if the other orientation will lead to a hollow heart rectangle
    # the steps below is to try turning a L-shape into a hollow rectangle 
    and_cropped_flipped = np.concatenate((cropped_180, cropped), axis=1)
    mirror_and_cropped = cv2.flip(and_cropped, -1)
    mirror_and_cropped_flipped = cv2.flip(and_cropped_flipped, -1)
    stacked = np.concatenate((and_cropped, mirror_and_cropped), axis=0)
    stacked_flipped = np.concatenate((and_cropped_flipped, mirror_and_cropped_flipped), axis=0)
    l, w = stacked.shape
    # flip the black and white pixels so we can find the contour of the hollow rectangle 
    inverted_stacked = (abs(stacked-1))
    inverted_stacked_flipped = (abs(stacked_flipped-1))
    # blur the image by a little bit so that as long as the overall shape roughly looks like a hollow rectangle it's ok
    inverted_stacked = cv2.blur(inverted_stacked, (10, 10))
    inverted_stacked_flipped = cv2.blur(inverted_stacked_flipped, (10, 10))
    image = inverted_stacked.astype('uint8')
    image_f = inverted_stacked_flipped.astype('uint8')
    cnt, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_f, _ = cv2.findContours(image_f, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if there are more than one contours, it is definitely not a hollow rectangle 
    if len(cnt) != 1 and len(cnt_f) != 1: 
        return False
    approx = cv2.approxPolyDP(cnt[0], 0.02*cv2.arcLength(cnt[0], True), True)
    approx_f = cv2.approxPolyDP(cnt_f[0], 0.02*cv2.arcLength(cnt_f[0], True), True)
    # see if the contour in the middle looks like a rectangle 
    if len(approx) or len(approx_f) == 4:
        return True 
    return False


'''Describe aspect ratio: narrow / medium-width / wide'''
def aspectRatio( mbr, description):
    # aspect Ratio is determined by the length of the smaller side of the rectangle over the length of the longer side of the rectangle
    for k, val in mbr.items():
        box = val[0]
        l = math.dist(box[0], box[1])
        w = math.dist(box[3], box[0])
        # choose the smaller one out of the two as width 
        ratio = min(l, w)/max(l, w)
        if ratio < NARROW: 
            description[k].append("narrow")
        elif ratio >= NARROW and ratio < MED_WIDTH:
            description[k].append("medium-width")
        else: 
            description[k].append("wide")


'''Describe size: smallest / small / medium-size / large / largest'''
def size(area, mbr, description):
    # size is determined by the length of the diagonal 
    smallest = (float('inf'), 0)
    largest = (float('-inf'), 0)
    for k, val in area.items():
        if val > largest[0]: 
            largest = (val, k) 
        if val < smallest[0]: 
            smallest = (val, k) 
        diagonal = mbr[k][1]
        if diagonal < MED_DIAGONAL: 
            description[k] = ["small"]
        elif diagonal >= MED_DIAGONAL and diagonal < LG_DIAGONAL:
            description[k] = ["medium-size"]
        else: 
            description[k] = ["large"]
    description[smallest[1]] = ["smallest"]
    description[largest[1]] = ["largest"]
    

'''Step 3: "Where"'''
def describeAbsSpace(mbr, description):
    im = cv2.imread(CAMPUS)
    determineVertHori(mbr, description, im.shape[0:2])
    for k, val in mbr.items():
        determineOrientation(k, val, description)
    

'''Helper function to determine verticality: uppermost / upper / mid-height / lower / lowermost & horizontality: leftmost / left / mid-width / right / rightmost'''
def determineVertHori(mbr, description, shape):
    # use the boundary box of the MBR to check for verticality and horizontality   
    # divide the image vertically by 5
    uppermost = shape[0] // 30 
    upper = 9 * shape[0] // 30
    midheight = 19 * shape[0] // 30
    lower = 29 * shape[0] // 30 
    # divide the image horizontally by 5
    leftmost = shape[1] // 30
    left = 9 * shape[1] // 30
    midwidth = shape[1] * 19 // 30
    right = 29 * shape[1] // 30
    for k, val in mbr.items():
        box = val[0]
        li = []
        li2 = []
        for ele in box: 
            pt = ele[1]
            pt2 = ele[0]
            # check which category the building belongs to vertically
            if pt < uppermost: 
                li.append('uppermost')
            elif pt >= uppermost and pt <= upper: 
                li.append('upper')
            elif pt > upper and pt <= midheight: 
                li.append('mid-height')
            elif pt > midheight and pt <= lower: 
                li.append('lower')
            else: 
                li.append('lowermost')
            # check which category the building belongs to horizontally 
            if pt2 < leftmost: 
                li2.append('leftmost')
            elif pt2 >= leftmost and pt2 <= left: 
                li2.append('left')
            elif pt2 > left and pt2 <= midwidth: 
                li2.append('mid-width')
            elif pt2 > midwidth and pt2 <= right: 
                li2.append('right')
            else: 
                li2.append('rightmost')
        added = []
        if 'uppermost' in li: 
            description[k].append('uppermost')
            added.append('vert')
        elif 'lowermost' in li: 
            description[k].append('lowermost')
            added.append('vert')
        if 'leftmost' in li2:
            description[k].append('leftmost')
            added.append('hori')
        elif 'rightmost' in li2:
            description[k].append('rightmost')
            added.append('hori')
        # if the building doesn't have a complete description yet 
        if len(added) < 2:
            if len(added) > 0 and added[0] == 'vert':
                data = Counter(li2)
                description[k].append(data.most_common(1)[0][0])
            elif len(added) > 0 and added[0] == 'hori':
                data = Counter(li)
                description[k].append(data.most_common(1)[0][0])
            else:
                data = Counter(li)
                description[k].append(data.most_common(1)[0][0])
                data = Counter(li2)
                description[k].append(data.most_common(1)[0][0])


'''Helper function to determine orientation: vertically-oriented / non-oriented / horizontally-oriented'''
def determineOrientation(k, mbr, description):
    # this is done by checking how similar the length and width of the mbr are to each other 
    # if the difference between l and w is signficant enough, then the building has an orientation 
    box = mbr[0]
    l = math.dist(box[0], box[1])
    w = math.dist(box[3], box[0])
    # if the building is rectangular, it has to have an orientation
    if 'rectangular' in description[k]:
        if l > w: 
            description[k].append('horizontally-oriented')
        else:
            description[k].append('vertically-oriented')
        return 
    if abs(l - w) / min(l, w) < ORIENTATION:
        description[k].append('non-oriented')
    elif l > w: 
        description[k].append('horizontally-oriented')
    else:
        description[k].append('vertically-oriented')


'''Step 4: "How" -- relative space'''
def relativeSpace(area, mbr_intersect, contours, t, sourceOrTarget):
    li = {}
    for i in t.keys():
        li[i] = []
        for j in t.keys():
            if i == j: 
                continue 
            re = None
            if sourceOrTarget == 'target':
                re = nearTo(j, i, area, mbr_intersect, contours[i], contours[j], t)
            elif sourceOrTarget == 'source':
                re = nearTo(i, j, area, mbr_intersect, contours[i], contours[j], t)
            if re: 
                li[i].append(t[j])
    return li
     

'''Helper function to check if T is near to S'''
def nearTo(S, T, area, mbr_intersect, cntS, cntT, t): 
    # if the buildings' mbr intersect, it means they are close to each other 
    if t[T] in mbr_intersect[S]:
        return True 
    #find the shortest distance between two pixels in the contour and that will be the minimum distance 
    min_d = float('inf')
    for s in cntS: 
        for t in cntT:
            d = math.dist(s[0], t[0])
            if d < min_d:
                min_d = d
    if 10*min_d/area[S] > 0.15:
        return False 
    return True
             

table_dict, table_dict2 = processTableTxt()
campus_im = cv2.imread(CAMPUS)
label_im = cv2.imread(LABEL)
area, com, mbr, mbr_intersect, contours = printRawData(table_dict, campus_im, label_im)
description = {}
describeShape(table_dict, area, com, mbr, mbr_intersect, description, contours)
re_target = relativeSpace(area, mbr_intersect, contours, table_dict, 'target')
re_source = relativeSpace(area, mbr_intersect, contours, table_dict, 'source')
describeAbsSpace(mbr, description)

# STEP 2 OUTPUT: 
# for k, val in description.items(): 
#     print(table_dict[k], val)
#     print('\tbuildings with same description')
#     print('\t',[table_dict[key] for key, v in description.items() if v == val and key != k])

# STEP 3 OUTPUT:
# for k, val in description.items(): 
#     print(table_dict[k], val[3:])
#     print('\tbuildings with same description')
#     print('\t',[table_dict[key] for key, v in description.items() if set(v[3:]) == set(val[3:]) and key != k])

# STEP 4 OUTPUT: 
# for k, val in re_source.items():
#     print(table_dict[k])
#     print('\tif it is source: ',val)
#     print('\tif it is target: ',re_target[k])
close = {}
# calculate which landmark source building is close to each building 
for k, val in re_target.items():
    maxx = (float('-inf'), None)
    for ele in val: 
        if len(re_source[table_dict2[ele]]) > maxx[0]:
            maxx = (len(re_source[table_dict2[ele]]), ele)
    close[k] = maxx[1]
# STEP 5 OUTPUT: 
# for k, val in description.items():
#     print(table_dict[k])
#     print('\t',val[0],',',val[1], ', and', val[2], 'structure located on the', val[3], 'and', val[4], 'side of campus and', val[5])
#     val = description[table_dict2[close[k]]]
#     print('\tit is close to',close[k],'which is a', val[0],',',val[1], ', and', val[2], 'structure located on the', val[3], 'and', val[4], 'side of campus and', val[5])