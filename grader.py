import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Grader:
    def __init__(self, img, test_answers, max_score=10):
        width, height = 1240, 1754 #fix định dạng ảnh thành 150PPI A4
        self.img = cv.resize(img, (width, height))
        self.test_answers = test_answers
        self.max_score = max_score
        self.saved_score = []
        self.result_img = img.copy()
    def __preprocess_contour(self, img):
        """Xử lý ảnh để detect contour các anchor points."""
        img = cv.medianBlur(img,3)
        img = cv.GaussianBlur(img,(5,5),0)
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 51, 5)
        return img
    def __preprocess_detect(self, img):
        """Xử lý ảnh để detect trắc nghiệm."""
        img = cv.GaussianBlur(img, (5, 5), 0)
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 51, 5)
        return img
    def __preprocess_edge(self, img):
        """Xử lý ảnh để lấy edge."""
        img = cv.GaussianBlur(img,(5,5),0)
        img = cv.Canny(img, threshold1=20, threshold2=100)
        return img
    def __four_point_transform(self, image, pts):
        """
            Hàm warp ảnh dùng 4 điểm.

            image: Ảnh ban đầu.
            pts: Array/List 2D gồm bốn điểm, mỗi điểm hai tọa độ [x,y].
        """
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    def __two_point_transform(self, image, pts, width):
        """
            Hàm warp ảnh dùng 2 điểm và chiều ngang tùy chọn.

            image: Ảnh ban đầu.
            pts: Array/List 2D gồm hai điểm, mỗi điểm hai tọa độ [x,y].
            width: Chiều ngang tùy chọn.
        """
        rect = np.zeros((4, 2), dtype = "float32")
        if pts[1][1] > pts[0][1]:
            rect[0] = pts[0] # top-left
            rect[3] = pts[1] # bottom-left
        else:
            rect[0] = pts[1] # top-left
            rect[3] = pts[0] # bottom-left
        rect[1] = rect[0] + [width, 0] # top-right
        rect[2] = rect[3] + [width, 0] # bottom-right
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    def __get_contours_rect(self, img, min_width=10,max_width=100, min_height=10,max_height=100, min_white_pixels=200):
        """
            Hàm detect contour (viền) hình vuông.
            Return các contour đc sắp xếp theo thứ tự diện tích từ lớn đến bé.

            img: Ảnh đầu vào.
            min_width, max_width, min_height, max_height: Chiều dài rộng tối thiểu và tối đa để thuật toán dectect.
            min_white_pixels: Số điểm pixel màu trắng tối thiểu cần có trong hình vuông.
        """
        ct = cv.findContours(img, cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
        contours, hierarchy = ct
        rectangles = []
        for contour in contours:
            epsilon = 0.03 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                x, y, w, h = cv.boundingRect(contour)
                aspect_ratio = float(w) / h
                if (aspect_ratio >= 0.7 and aspect_ratio <= 1.2 and
                w >= min_width and w <= max_width and
                h >= min_height and h <= max_height):
                    roi = img[y:y+h, x:x+w]
                    white_pixel_count = np.sum(roi > 100)
                    if white_pixel_count >= min_white_pixels:
                        rectangles.append(contour)
        rectangle_areas = [(rect, cv.contourArea(rect)) for rect in rectangles]
        sorted_rectangle_areas = sorted(rectangle_areas, key=lambda x: x[1], reverse=True)
        rectangles = [rect_area[0] for rect_area in sorted_rectangle_areas]
        return rectangles
    def __get_contours_circle(self, img, min_radius = 12, max_radius = 20, min_white_pixels = None):
        """
            Hàm detect contour (viền) hình tròn.
            Return các contour đc sắp xếp theo thứ tự diện tích từ lớn đến bé.

            img: Ảnh đầu vào.
            min_radius: Bán kính tối thiểu để thuật toán dectect.
            max_radius: Bán kính tối đa để thuật toán dectect.
        """
        ct = cv.findContours(img, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = ct
        circles = []
        for contour in contours:
            epsilon = 0.03 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            if len(approx) >= 8:
                (x, y), radius = cv.minEnclosingCircle(contour)
                center = (int(round(x)), int(round(y)))
                radius = int(radius)
                if radius >= min_radius and radius <= max_radius:
                    if min_white_pixels != None:
                        roi = img[int(round(y-radius)):int(round(y+radius)),int(round(x-radius)):int(round(x+radius))]
                        white_pixel_count = np.sum(roi > 100)
                        if white_pixel_count >= min_white_pixels:
                            circles.append((center, radius))
                    else: circles.append((center, radius))
        circle_areas = [(circle, np.pi*radius**2) for circle, radius in circles]
        sorted_circle_areas = sorted(circle_areas, key=lambda x: x[1], reverse=True)
        circles = [circle_area[0] for circle_area in sorted_circle_areas]
        circles_array = np.unique(np.asarray(list(map(list, circles))),axis=0)
        return circles_array
    def __get_anchor(self, contour, num_anchors = 4):
        """
        Hàm tính trung bình các tọa độ viền.
        Trả về các điểm anchor cần thiết để warp ảnh.

        contour: Tọa độ viền đã dectect.
        num_anchors: Số điểm anchor. Phải bằng số hình vuông/tròn có trong contour.
        """
        anchor_x = []
        anchor_y = []
        for i in range(len(contour)):
            x = int(round(np.mean(contour[i][:, :, 0])))
            y = int(round(np.mean(contour[i][:, :, 1])))
            anchor_x.append(x)
            anchor_y.append(y)
        temp_anchors, idx = np.unique(np.column_stack((anchor_x,anchor_y)),axis=0,return_index=True)
        anchors = temp_anchors[np.argsort(idx)]
        return anchors[:num_anchors]
    def __group_avg_values(self, sorted_array, offset=15):
        """
            Hàm dùng để fix trường hợp không thể detect tất cả các contour hình tròn hoặc detect trùng tọa độ trong sheet đáp án/sbd/mã đề.

            sorted_array: Array/List 1D đã sort của x hoặc y.
            offset: Chênh lệch tối đa giữa các điểm để được tính vào làm một nhóm.
        """
        groups = []
        current_group = [sorted_array[0]]
        for i in range(1, len(sorted_array)):
            if abs(sorted_array[i] - current_group[-1]) <= offset:
                current_group.append(sorted_array[i])
            else:
                groups.append(int(round(sum(current_group)/len(current_group))))
                current_group = [sorted_array[i]]
        groups.append(int(round(sum(current_group)/len(current_group))))
        return groups
    def __matrix2ans(self, np_matrix, threshold=20000, start_index=1):
        """
            Hàm chuyển đổi từ ma trận sang kết quả ABCD.

            np_matrix: Ma trận 2D numpy.
            threshold: Độ sáng tối thiểu để được tính kết quả.
            start_index: Vị trí câu bắt đầu.
        """
        ans = {}
        for row in range(np_matrix.shape[0]):
            temp = []
            for col in range(np_matrix.shape[1]):
                if np_matrix[row][col] > threshold:
                    temp.append(col)
            ans.update({row+start_index:",".join([self.__decode(num) for num in temp])})
        return ans
    def __find_correct_ans(self, user_answers, test_answers):
        """
            Tìm kết quả đúng.
            Trả về dictionary {vị trí câu: kết quả}.

            user_answers: Câu trả lời cần kiểm tra.
            test_answers: Câu trả lời đúng.
        """
        correct_answers = {k: test_answers[k] for k in test_answers if k in user_answers and test_answers[k] == user_answers[k]}
        return correct_answers
    def __decode(self, num):
        """
            Chuyển từ 0->3 sang A->D.
        """
        if num == 0: return 'A'
        elif num == 1: return 'B'
        elif num == 2: return 'C'
        elif num == 3: return 'D'
        else: return 'NAN'
    def __get_correct_answers(self, img_answers, test_answers, get_raw_answers=False):
        """
            Lấy kết quả đúng.
        """
        student_correct_answers = {}
        student_answers = {}
        for i in range(len(img_answers)):
            img = self.__preprocess_edge(img_answers[i])
            img_detect = self.__preprocess_detect(img_answers[i])
            circles = self.__get_contours_circle(img)
            # Fix undetected circles
            x_sorted = np.sort(circles[:,:1].reshape(-1)) # sort x
            y_sorted = np.sort(circles[:,1:2].reshape(-1)) # sort y
            x_group = self.__group_avg_values(x_sorted, offset=15)
            y_group = self.__group_avg_values(y_sorted, offset=15)

            white_pixel_counts = []
            min_white_pixels = 100
            radius = 15
            rows = len(y_group)
            cols = len(x_group)
            start_index = 1

            for y in y_group:
                for x in x_group:
                    center = (x,y)
                    roi = img_detect[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius]
                    # Count the white pixels in the roi
                    count = np.sum(roi)
                    white_pixel_counts.append(count)
            if i == 0: start_index=1
            if i == 1: start_index=18
            if i == 2: start_index=35
            white_pixel_counts = np.asarray(white_pixel_counts).reshape((rows,cols))
            white_pixel_counts = white_pixel_counts-np.average(white_pixel_counts)
            ans = self.__matrix2ans(white_pixel_counts, threshold = 15000 , start_index=start_index)
            print(ans)
            student_correct_answers.update(self.__find_correct_ans(ans, test_answers))
            student_answers.update(ans)
        if get_raw_answers == True:
            return student_correct_answers, student_answers
        else:
            return student_correct_answers
    def __encode_ABCD(self, input_):
        if len(input_) == 1:
            if input_ == 'A': return [1] # Trả về list để đồng nhất với TH len(input_)!=1
            elif input_ == 'B': return [2]
            elif input_ == 'C': return [3]
            elif input_ == 'D': return [4]
            else: raise Exception("Chỉ nhận ABCD ở hàm encode_ABCD!")
        elif len(input_) > 1:
            input_ = input_.replace(' ', '').split(",")
            return [self.__encode_ABCD(ans)[0] for ans in input_ ]
        else:
            raise Exception("Chỉ nhận ABCD ở hàm encode_ABCD! Nhận được input rỗng.")
    def __get_info(self, img_info):
        """
            Lấy thông tin người làm bài.
        """
        student_info = []
        threshold = 20000
        for i in range(len(img_info)):
            img = self.__preprocess_edge(img_info[i])
            img_detect = self.__preprocess_detect(img_info[i])
            circles = self.__get_contours_circle(img)
            # Fix undetected circles
            x_sorted = np.sort(circles[:,:1].reshape(-1)) # sort x
            y_sorted = np.sort(circles[:,1:2].reshape(-1)) # sort y
            x_group = self.__group_avg_values(x_sorted, offset=10)
            y_group = self.__group_avg_values(y_sorted, offset=10)

            white_pixel_counts = []
            min_white_pixels = 100
            radius = 5
            rows = len(y_group)
            cols = len(x_group)

            for y in y_group:
                for x in x_group:
                    center = (x,y)
                    roi = img_detect[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius]
                    # Count the white pixels in the roi
                    count = np.sum(roi)
                    white_pixel_counts.append(count)
            white_pixel_counts = np.asarray(white_pixel_counts).reshape((rows,cols))
            white_pixel_counts = white_pixel_counts-(np.average(white_pixel_counts)*0.8)
            if white_pixel_counts.shape[1] == 3 and sum(sum(white_pixel_counts>threshold)) != 3:
                raise Exception("Mã đề không hợp lệ!")
            if white_pixel_counts.shape[1] == 6 and sum(sum(white_pixel_counts>threshold)) != 6:
                raise Exception("Số báo danh không hợp lệ!")
            detected_info = "".join(map(str,np.argmax(white_pixel_counts, axis=0)))
            student_info.append(detected_info)
        return student_info
    def get_info(self):
        img_origin = self.img.copy()
        img_pre = self.__preprocess_contour(self.img.copy())

        #detect contours của 12 anchors
        rectangles = self.__get_contours_rect(img_pre)
        #Lấy 4 anchor points để chuẩn bị warp
        warp_anchor_img = self.__get_anchor(rectangles, num_anchors = 4)
        center = np.mean(warp_anchor_img, axis=0)
        for i in range(len(warp_anchor_img)): # Cắt bớt phần dư
            if warp_anchor_img[i][0] <= center[0]:
                warp_anchor_img[i][0] += 25
            else:
                warp_anchor_img[i][0] -= 25
        #Warp vào 4 anchor points
        img_warp = self.__four_point_transform(img_origin.copy(),warp_anchor_img)
        img_pre_warp = self.__four_point_transform(img_pre.copy(),warp_anchor_img)

        #Lấy tiếp các anchor points ở sbd, mã đề và trắc nghiệm để chuẩn bị warp part 2
        small_rectangles = self.__get_contours_rect(img_pre_warp, min_width=20, min_height=20, min_white_pixels=400)
        warp_anchor_in = self.__get_anchor(small_rectangles, len(small_rectangles))

        if len(warp_anchor_in)==0 | len(warp_anchor_in)<8:
            raise Exception("Vui lòng chụp lại ảnh!")

        width = img_warp.shape[1]
        height = img_warp.shape[0]
        upper_anchor = []
        lower_anchor = []
        warp_anchor_in = warp_anchor_in[np.argsort(warp_anchor_in[:,:1].reshape(-1), axis=0)] #sort theo x
        for pts in warp_anchor_in:
            if pts[1] < height*(1/3): # phân ra anchor nửa trên (cho mã đề, số báo danh) và nửa dưới (cho trắc nghiệm)
                upper_anchor.append(pts)
            else:
                lower_anchor.append(pts)
        upper_anchor = np.asarray(upper_anchor).reshape(-1,2,2) #reshape (group, 2 pairs, 2 values xy)
        lower_anchor = np.asarray(lower_anchor).reshape(-1,2,2)
        if len(upper_anchor[0])%2 != 0 | len(upper_anchor[0]) == 0 | len(lower_anchor[0]) == 0:
            raise Exception("Vui lòng chụp lại ảnh!")
        img_info = []
        if len(upper_anchor) == 2:
            sorted_upper_anchor = np.array([sorted(group, key=lambda x: x[1]) for group in upper_anchor])
            distance = np.abs(np.diff(sorted_upper_anchor[:,:1,:1], axis=0))
            avg_distance = int(round(np.mean(distance)))
            offset=15
            img_info.append(self.__two_point_transform(img_warp,upper_anchor[0], width=avg_distance-offset))
            img_info.append(self.__two_point_transform(img_warp,upper_anchor[1], width=(avg_distance/2)))
        elif len(upper_anchor) == 1:
            img_info.append(self.__two_point_transform(img_warp,upper_anchor[0], width=150))
        else: raise Exception("Vui lòng chụp lại ảnh!")
        student_info = self.__get_info(img_info)
        return student_info

    def grade(self):
        """
            Hàm chấm điểm.
            Trả về các kết quả đúng, thông tin người thi và số điểm.

            img: Ảnh bài thi.
            test_answers: Kết quả đúng.
        """
        img_origin = self.img.copy()
        img_pre = self.__preprocess_contour(self.img.copy())
        show_true_answer = False

        #detect contours của 12 anchors
        rectangles = self.__get_contours_rect(img_pre)
        #Lấy 4 anchor points để chuẩn bị warp
        warp_anchor_img = self.__get_anchor(rectangles, num_anchors = 4)
        center = np.mean(warp_anchor_img, axis=0)
        for i in range(len(warp_anchor_img)): # Cắt bớt phần dư
            if warp_anchor_img[i][0] <= center[0]:
                warp_anchor_img[i][0] += 25
            else:
                warp_anchor_img[i][0] -= 25
        #Warp vào 4 anchor points
        img_warp = self.__four_point_transform(img_origin.copy(),warp_anchor_img)
        img_pre_warp = self.__four_point_transform(img_pre.copy(),warp_anchor_img)
        img_warp_color = cv.cvtColor(img_warp, cv.COLOR_GRAY2BGR)

        #Lấy tiếp các anchor points ở sbd, mã đề và trắc nghiệm để chuẩn bị warp part 2
        small_rectangles = self.__get_contours_rect(img_pre_warp, min_width=20, min_height=20, min_white_pixels=400)
        warp_anchor_in = self.__get_anchor(small_rectangles, len(small_rectangles))

        if len(warp_anchor_in)==0 | len(warp_anchor_in)<8:
            raise Exception("Vui lòng chụp lại ảnh!")

        width = img_warp.shape[1]
        height = img_warp.shape[0]
        upper_anchor = []
        lower_anchor = []
        warp_anchor_in = warp_anchor_in[np.argsort(warp_anchor_in[:,:1].reshape(-1), axis=0)] #sort theo x
        for pts in warp_anchor_in:
            if pts[1] < height*(1/3): # phân ra anchor nửa trên (cho mã đề, số báo danh) và nửa dưới (cho trắc nghiệm)
                upper_anchor.append(pts)
            else:
                lower_anchor.append(pts)
        upper_anchor = np.asarray(upper_anchor).reshape(-1,2,2) #reshape (group, 2 pairs, 2 values xy)
        lower_anchor = np.asarray(lower_anchor).reshape(-1,2,2)
        if len(upper_anchor[0])%2 != 0 | len(upper_anchor[0]) == 0 | len(lower_anchor[0]) == 0:
            raise Exception("Vui lòng chụp lại ảnh!")

        img_answers = []
        sorted_lower_anchor = np.array([sorted(group, key=lambda x: x[1]) for group in lower_anchor])
        distance = np.abs(np.diff(sorted_lower_anchor[:,:1,:1], axis=0))
        avg_distance = int(round(np.mean(distance)))
        offset=20
        for i in range(len(lower_anchor)):
            img_answers.append(self.__two_point_transform(img_warp,lower_anchor[i], width=avg_distance-offset))

        img_info = []
        if len(upper_anchor) == 2:
            sorted_upper_anchor = np.array([sorted(group, key=lambda x: x[1]) for group in upper_anchor])
            distance = np.abs(np.diff(sorted_upper_anchor[:,:1,:1], axis=0))
            avg_distance = int(round(np.mean(distance)))
            offset=15
            img_info.append(self.__two_point_transform(img_warp,upper_anchor[0], width=avg_distance-offset))
            img_info.append(self.__two_point_transform(img_warp,upper_anchor[1], width=(avg_distance/2)))
        elif len(upper_anchor) == 1:
            img_info.append(self.__two_point_transform(img_warp,upper_anchor[0], width=150))
        else: raise Exception("Vui lòng chụp lại ảnh!")

        student_correct_ans, student_raw_answers = self.__get_correct_answers(img_answers, self.test_answers, get_raw_answers=True)
        student_info = self.__get_info(img_info)
        score = (len(student_correct_ans)/len(self.test_answers))*self.max_score

        #Phân Group
        result_img = img_warp_color.copy()
        for idx, ans in self.test_answers.items():
            if idx < 18: 
                group = 0
                row = idx
            elif idx < 35: 
                group = 1
                row = idx-17
            else: 
                group = 2
                row = idx-34
            if student_raw_answers[idx] == ans:# Nếu trả lời đúng
                encoded_ABCD = self.__encode_ABCD(ans)
                cols = encoded_ABCD #cols vì hàm encode_ABCD có thể trả ra nhiều hơn một số nếu user khoanh 2 đáp án cùng lúc.
                text = ans
                main_color = (0, 255, 0) #green
            else:# Trả lời sai
                encoded_ABCD = self.__encode_ABCD(student_raw_answers[idx])
                cols = encoded_ABCD
                text = student_raw_answers[idx]
                main_color = (0, 0, 255) #red
                # Hiển thị đáp án đúng
                # if show_true_answer == True:
                #     true_col =  self.__encode_ABCD(ans)
                #     true_x_offset_circle = round(((sorted_lower_anchor[1][0][0]-sorted_lower_anchor[0][0][0])/6+
                #                         (sorted_lower_anchor[1][1][0]-sorted_lower_anchor[0][1][0])/6)/2) * true_col  
                #     cv.circle(result_img,
                #             ((sorted_lower_anchor[group][0][0]+sorted_lower_anchor[group][1][0])//2 + true_x_offset_circle, 
                #             sorted_lower_anchor[group][0][1]+round(abs(sorted_lower_anchor[group][0][1]-sorted_lower_anchor[group][1][1])/18.2)*row+5),
                #             radius=5,
                #             color=(0, 255, 0), 
                #             thickness=10)
            for col in cols:
                x_offset_text = -35
                x_offset_circle = round(((sorted_lower_anchor[1][0][0]-sorted_lower_anchor[0][0][0])/6+
                                        (sorted_lower_anchor[1][1][0]-sorted_lower_anchor[0][1][0])/6)/2) * col  
                cv.putText(result_img, text,
                            org=((sorted_lower_anchor[group][0][0]+sorted_lower_anchor[group][1][0])//2 + x_offset_text, 
                            sorted_lower_anchor[group][0][1]+round(abs(sorted_lower_anchor[group][0][1]-sorted_lower_anchor[group][1][1])/18)*row+10),
                            color=main_color, 
                            thickness=2,
                            fontScale=1,
                            fontFace=cv.LINE_AA)
                cv.circle(result_img,
                            ((sorted_lower_anchor[group][0][0]+sorted_lower_anchor[group][1][0])//2 + x_offset_circle, 
                            sorted_lower_anchor[group][0][1]+round(abs(sorted_lower_anchor[group][0][1]-sorted_lower_anchor[group][1][1])/18.2)*row+5),
                            radius=5,
                            color=main_color, 
                            thickness=10)
        return student_correct_ans, student_info, score, result_img