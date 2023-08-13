import cv2


class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        # Convert Image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # mask1 = cv2.GaussianBlur(gray.copy(), (5,5), 0)


        #performing different edge detection methods to compare them
        gradients_sobelx = cv2.Sobel(gray, -1, 1, 0)
        gradients_sobely = cv2. Sobel(gray, -1, 0, 1)
        gradients_sobelxy = cv2.addWeighted(gradients_sobelx, 0.5, gradients_sobely, 0.5, 0)

        gradients_laplacian = cv2.Laplacian(gray, -1)

        canny = cv2.Canny(gray, 10, 150)


        # threshold the image
        _ ,thresh = cv2.threshold(gradients_sobelxy, 18, 255, cv2.THRESH_BINARY)

        # construct a closing kernel and apply it to the thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # perform a series of erosions and dilations
        closed = cv2.erode(closed, None, iterations = 2)
        closed = cv2.dilate(closed, None, iterations = 2)

        
        # Find contours
        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # draw contours
        # contours_img = cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)

        # showing all different edge detection techniques
        # cv2.imshow("original", gray)
        # # cv2.imshow("gaussian", mask1)
        # cv2.imshow("sobelxy", gradients_sobelxy)
        # # cv2.imshow("laplacian", gradients_laplacian)
        # cv2.imshow("canny", canny)
        # cv2.imshow("thresh", thresh)
        # cv2.imshow("closed", closed)
        
        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                # cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                objects_contours.append(cnt)

        return objects_contours

    # def get_objects_rect(self):
    #     box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    #     box = np.int0(box)

# img = cv2.imread("phone1.jpg")
# detector = HomogeneousBgDetector()
# detector.detect_objects(img)
# cv2.waitKey()
