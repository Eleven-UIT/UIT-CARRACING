from client_lib import GetStatus, GetRaw, GetSeg, AVControl, CloseSocket
import cv2
import numpy as np
import math

# Tham số
CHECKPOINT = 160   # dòng checkpoint (chỉnh tùy ảnh cao/thấp)
MAX_ANGLE = 25     # giới hạn góc đánh lái
BASE_SPEED = 22    # tốc độ cơ bản

def AngCal(seg_img):
    """
    Tính toán góc lái từ ảnh segmentation.
    Dùng 1 dòng checkpoint để tìm tâm làn đường.
    """
    gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    gray = (gray * (255 / np.max(gray))).astype(np.uint8)

    h, w = gray.shape

    # Lấy một hàng pixel ở CHECKPOINT
    line_row = gray[CHECKPOINT, :]

    flag = True
    min_x, max_x = 0, 0
    
    for x, y in enumerate(line_row):
        if y == 255 and flag:
            flag = False
            min_x = x
        elif y == 255:
            max_x = x

    # Nếu không tìm thấy lane → fallback lane giữa
    if max_x == 0 and min_x == 0:
        return 0.0  

    center_lane = int((max_x + min_x) / 2)

    # Tính góc bằng hình học
    x0, y0 = int(w/2), h         # tâm xe
    x1, y1 = center_lane, CHECKPOINT  # tâm lane tại checkpoint

    slope = (x1 - x0) / (y0 - y1 + 1e-5)
    angle = math.degrees(math.atan(slope))

    # Giới hạn góc lái
    angle = max(-MAX_ANGLE, min(MAX_ANGLE, angle))
    return angle


if __name__ == "__main__":
    try:
        while True:
            state = GetStatus()
            seg_img = GetSeg()

            if seg_img is None:
                continue

            # Tính góc lái
            angle = AngCal(seg_img)
            speed = BASE_SPEED

            # Gửi điều khiển
            AVControl(speed=speed, angle=angle)

            # Debug hiển thị
            h, w, _ = seg_img.shape
            cx = int(w/2 + math.tan(math.radians(angle)) * (h - CHECKPOINT))
            debug_img = seg_img.copy()
            cv2.line(debug_img, (w//2, h), (cx, CHECKPOINT), (0,255,0), 2)
            cv2.imshow("Lane Debug", debug_img)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print("Closing socket")
        CloseSocket()
        cv2.destroyAllWindows()
