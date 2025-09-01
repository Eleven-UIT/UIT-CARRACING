from client_lib import GetStatus, GetSeg, AVControl, CloseSocket
import cv2
import numpy as np
import time
from simple_pid import PID

# =========================
# Config
# =========================
class Config:
    UPPER_CHECKPOINT = 150
    MID_CHECKPOINT = 130
    LOWER_CHECKPOINT = 120

    MAX_STEERING_ANGLE = 30
    MAX_VEHICLE_SPEED = 40
    MIN_VEHICLE_SPEED = 1
    SLOW_DOWN_ANGLE = 17

    WHITE_PIXEL_VALUE = 255
    CENTER_BIAS = 0.58
    LINE_WIDTH = 2

    # Angle mapping
    ANGLE_INPUT_RANGE = (-5, 5)
    ANGLE_OUTPUT_RANGE = (-25, 25)

    # PID tuning sets
    PID_STRAIGHT = (0.8, 0.01, 0.3)   # đường thẳng
    PID_CURVE    = (1.4, 0.02, 0.5)   # cua / zig-zag
    STRAIGHT_SLOPE_THRESHOLD = 0.2


# =========================
# Kalman Filter setup
# =========================
kalman = cv2.KalmanFilter(2, 1)
kalman.measurementMatrix = np.array([[1, 0]], np.float32)
kalman.transitionMatrix  = np.array([[1, 1], [0, 1]], np.float32)
kalman.processNoiseCov   = np.array([[1, 0], [0, 1]], np.float32) * 0.03


# =========================
# PID Controller setup
# =========================
pid_controller = PID(*Config.PID_STRAIGHT, setpoint=0)
pid_controller.output_limits = (-Config.MAX_STEERING_ANGLE, Config.MAX_STEERING_ANGLE)


def set_pid_tunings(pid: PID, kp, ki, kd):
    pid.tunings = (kp, ki, kd)


# =========================
# Image processing
# =========================
def process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (gray_image * (Config.WHITE_PIXEL_VALUE / np.max(gray_image))).astype(np.uint8)


def find_line_centers(image, checkpoints):
    centers = []
    for cp in checkpoints:
        pixel_row = image[cp, :]
        white_pixels = np.where(pixel_row == Config.WHITE_PIXEL_VALUE)[0]
        if len(white_pixels) == 0:
            centers.append(image.shape[1] // 2)
        else:
            min_x, max_x = white_pixels[0], white_pixels[-1]
            if max_x - min_x >= image.shape[1] - 1:
                centers.append(image.shape[1] // 2)
            else:
                centers.append(int((min_x + max_x + 1) * Config.CENTER_BIAS))
    return centers


def derive_slope_and_angle(image):
    gray = process_image(image)
    h, w = gray.shape
    checkpoints = [Config.UPPER_CHECKPOINT, Config.MID_CHECKPOINT, Config.LOWER_CHECKPOINT]
    centers = find_line_centers(gray, checkpoints)

    slopes = [(c - (w // 2)) / (h - cp) for c, cp in zip(centers, checkpoints) if c != w // 2]
    avg_slope = np.mean(slopes) if slopes else 0

    mapped_angle = (avg_slope - Config.ANGLE_INPUT_RANGE[0]) * \
                   (Config.ANGLE_OUTPUT_RANGE[1] - Config.ANGLE_OUTPUT_RANGE[0]) / \
                   (Config.ANGLE_INPUT_RANGE[1] - Config.ANGLE_INPUT_RANGE[0]) + \
                   Config.ANGLE_OUTPUT_RANGE[0]

    return mapped_angle, avg_slope, gray, centers


# =========================
# Speed control
# =========================
def compute_vehicle_speed(steering_angle, last_angle):
    if abs(steering_angle - last_angle) > Config.SLOW_DOWN_ANGLE:
        return max(Config.MIN_VEHICLE_SPEED + 4, Config.MAX_VEHICLE_SPEED * 0.25)
    adjusted_speed = Config.MAX_VEHICLE_SPEED - abs(steering_angle) * \
                     (Config.MAX_VEHICLE_SPEED - Config.MIN_VEHICLE_SPEED) / Config.MAX_STEERING_ANGLE
    return max(Config.MIN_VEHICLE_SPEED, adjusted_speed)


# =========================
# Debug visualization
# =========================
def display_image(processed_image, centers, avg_slope):
    h, w = processed_image.shape
    for cp, c in zip([Config.UPPER_CHECKPOINT, Config.MID_CHECKPOINT, Config.LOWER_CHECKPOINT], centers):
        cv2.line(processed_image, (0, cp), (w, cp), 100, Config.LINE_WIDTH)
        cv2.circle(processed_image, (c, cp), 5, 200, -1)

    angle_end_x = int(w / 2 + (h - Config.UPPER_CHECKPOINT) * avg_slope)
    cv2.line(processed_image, (w // 2, h), (angle_end_x, Config.UPPER_CHECKPOINT), 50, Config.LINE_WIDTH + 3)


# =========================
# Main loop
# =========================
if __name__ == "__main__":
    try:
        last_angle = 0.0
        while True:
            time.sleep(0.02)

            _status = GetStatus()
            seg_img = GetSeg()
            if seg_img is None:
                AVControl(speed=Config.MIN_VEHICLE_SPEED, angle=0)
                continue

            mapped_angle, avg_slope, gray, centers = derive_slope_and_angle(seg_img)

            # Adaptive PID tuning
            if abs(avg_slope) < Config.STRAIGHT_SLOPE_THRESHOLD:
                set_pid_tunings(pid_controller, *Config.PID_STRAIGHT)
            else:
                set_pid_tunings(pid_controller, *Config.PID_CURVE)

            # Kalman smoothing
            try:
                _ = kalman.predict()
                kalman.correct(np.array([[np.float32(mapped_angle)]]))
                smoothed_angle = float(kalman.statePost[0, 0])
            except Exception:
                smoothed_angle = mapped_angle

            # PID
            try:
                pid_output = pid_controller(-smoothed_angle)
            except Exception:
                pid_output = -smoothed_angle

            # Speed control
            vehicle_speed = compute_vehicle_speed(mapped_angle, last_angle)
            last_angle = mapped_angle

            # 🚗 Print style Lightning McQueen
            if vehicle_speed >= Config.MAX_VEHICLE_SPEED * 0.9:
                print(f"KACHOW!!! 🚀 I am speed ⚡ {vehicle_speed:.2f}")
            else:
                print(f"I am speed ⚡ {vehicle_speed:.2f} | "
                      f"raw:{mapped_angle:+.2f} slope:{avg_slope:+.3f} smoothed:{smoothed_angle:+.2f} pid:{pid_output:+.2f}")

            # Send control
            AVControl(speed=vehicle_speed, angle=pid_output)

            # Debug window
            display_image(gray, centers, avg_slope)
            cv2.imshow('Lightning McQueen', cv2.resize(gray, (360, 270)))

            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        print(f"Runtime error: {e}")
    finally:
        CloseSocket()
        cv2.destroyAllWindows()
