from client_lib import GetStatus, GetSeg, AVControl, CloseSocket
import cv2
import numpy as np
import time

# ================== CONFIG ==================
class Config:
    LOOKAHEAD_CHECKPOINT = 100
    UPPER_CHECKPOINT = 160
    MID_CHECKPOINT = 140
    LOWER_CHECKPOINT = 120

    MAX_STEERING_ANGLE = 25
    SLOW_DOWN_ANGLE = 10

    MAX_VEHICLE_SPEED = 36   # tốc độ tối đa (rule giữ cao)
    MIN_VEHICLE_SPEED = 5

    WHITE_PIXEL_VALUE = 255
    CENTER_BIAS = 0.6
    LINE_WIDTH = 2

    ANGLE_INPUT_RANGE = (-5, 5)
    ANGLE_OUTPUT_RANGE = (-25, 25)


# ================== PID ==================
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def compute(self, target_value, current_value):
        error = target_value - current_value
        self.integral += error
        derivative = error - self.previous_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output


pid_controller = PIDController(1.2, 0.01, 0.08)


# ================== IMAGE PROCESSING ==================
def process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (gray_image * (Config.WHITE_PIXEL_VALUE / np.max(gray_image))).astype(np.uint8)


def apply_purple_colormap(gray_image):
    norm_img = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
    colored = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
    colored[:, :, 0] = norm_img * 0.8   # Blue
    colored[:, :, 1] = norm_img * 0.2   # Green
    colored[:, :, 2] = norm_img * 0.8   # Red
    return colored


def find_line_centers(image, checkpoints):
    centers = []
    for cp in checkpoints:
        pixel_row = image[cp, :]
        white_pixels = np.where(pixel_row == Config.WHITE_PIXEL_VALUE)[0]
        if len(white_pixels) == 0:
            centers.append(int(image.shape[1] / 2))
        else:
            min_x, max_x = white_pixels[0], white_pixels[-1]
            if max_x - min_x >= image.shape[1] - 1:
                centers.append(int(image.shape[1] / 2))
            else:
                centers.append(int((min_x + max_x + 1) * Config.CENTER_BIAS))
    return centers


def calculate_weighted_slope(center_positions, image_width, image_height):
    weights = [0.4, 0.3, 0.2, 0.1]  # thêm lookahead
    weighted_slopes = []
    midpoint = int(image_width / 2)

    for center, checkpoint, weight in zip(center_positions,
                                          [Config.LOOKAHEAD_CHECKPOINT, Config.UPPER_CHECKPOINT, Config.MID_CHECKPOINT, Config.LOWER_CHECKPOINT],
                                          weights):
        slope = (center - midpoint) / (image_height - checkpoint)
        weighted_slopes.append(slope * weight)

    return sum(weighted_slopes)


def compute_steering_angle(image):
    gray_image = process_image(image)
    image_height, image_width = gray_image.shape
    checkpoints = [Config.LOOKAHEAD_CHECKPOINT, Config.UPPER_CHECKPOINT, Config.MID_CHECKPOINT, Config.LOWER_CHECKPOINT]
    center_positions = find_line_centers(gray_image, checkpoints)

    avg_slope = calculate_weighted_slope(center_positions, image_width, image_height)
    steering_angle = (avg_slope - Config.ANGLE_INPUT_RANGE[0]) * \
                     (Config.ANGLE_OUTPUT_RANGE[1] - Config.ANGLE_OUTPUT_RANGE[0]) / \
                     (Config.ANGLE_INPUT_RANGE[1] - Config.ANGLE_INPUT_RANGE[0]) + Config.ANGLE_OUTPUT_RANGE[0]

    return steering_angle, gray_image, center_positions, avg_slope


# ================== RULE-BASED SPEED CONTROL ==================
def compute_vehicle_speed(steering_angle, last_angle):
    base_speed = Config.MAX_VEHICLE_SPEED

    # Nếu cua gắt thì giảm tốc mạnh
    if abs(steering_angle) > 18:
        adjusted_speed = 10
    elif abs(steering_angle) > 12:
        adjusted_speed = 16
    else:
        adjusted_speed = base_speed

    # Nếu đổi hướng đột ngột
    if abs(steering_angle - last_angle) > Config.SLOW_DOWN_ANGLE:
        adjusted_speed = max(adjusted_speed * 0.7, Config.MIN_VEHICLE_SPEED)

    return max(Config.MIN_VEHICLE_SPEED, adjusted_speed)


def display_image(processed_image, center_positions, avg_slope, vehicle_speed):
    height, width, _ = processed_image.shape

    for checkpoint, center in zip([Config.LOOKAHEAD_CHECKPOINT, Config.UPPER_CHECKPOINT, Config.MID_CHECKPOINT, Config.LOWER_CHECKPOINT], center_positions):
        cv2.line(processed_image, (0, checkpoint), (width, checkpoint), (100, 100, 100), Config.LINE_WIDTH)
        cv2.circle(processed_image, (center, checkpoint), 5, (200, 200, 200), -1)

    angle_end_x = int(width / 2 + (height - Config.UPPER_CHECKPOINT) * avg_slope)
    cv2.line(processed_image, (int(width / 2), height), (angle_end_x, Config.UPPER_CHECKPOINT), (50, 50, 200), Config.LINE_WIDTH + 3)

    cv2.putText(processed_image, f"Speed: {vehicle_speed:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 255), 2, cv2.LINE_AA)

    cv2.imshow('Lightning Mcqueen', cv2.resize(processed_image, (320, 240)))


# ================== MAIN LOOP ==================
if __name__ == "__main__":
    try:
        previous_angle = 0
        while True:
            time.sleep(0.02)

            status = GetStatus()
            segmented_image = GetSeg()

            if segmented_image is None:
                print("Không thể nhận diện hình ảnh phân đoạn.")
                continue

            steering_angle, gray_image, center_positions, avg_slope = compute_steering_angle(segmented_image)

            # Adaptive PID: tăng kp khi cua gắt
            if abs(steering_angle) > 15:
                pid_controller.kp = 1.6
                pid_controller.kd = 0.12
            else:
                pid_controller.kp = 1.2
                pid_controller.kd = 0.08

            pid_adjusted_angle = pid_controller.compute(0, -steering_angle)
            vehicle_speed = compute_vehicle_speed(steering_angle, previous_angle)
            previous_angle = steering_angle

            print(f"[DEBUG] Angle = {pid_adjusted_angle:.2f}°, Speed = {vehicle_speed:.2f}")

            purple_image = apply_purple_colormap(gray_image)
            display_image(purple_image, center_positions, avg_slope, vehicle_speed)

            AVControl(speed=vehicle_speed, angle=pid_adjusted_angle)

            if cv2.waitKey(1) == ord('q'):
                break
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        CloseSocket()
