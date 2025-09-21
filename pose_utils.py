import cv2
import mediapipe as mp
import numpy as np
import math


def set_pose_parameters():
	mode = False
	complexity = 1
	smooth_landmarks = True
	enable_segmentation = False
	smooth_segmentation = True
	detectionCon = 0.5
	trackCon = 0.5
	mpPose = mp.solutions.pose
	return mode,complexity,smooth_landmarks,enable_segmentation,smooth_segmentation,detectionCon,trackCon,mpPose


def get_angle(img, landmark_list, point1, point2, point3, draw=True):
	x1, y1 = landmark_list[point1][1:]
	x2, y2 = landmark_list[point2][1:]
	x3, y3 = landmark_list[point3][1:]
	angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
	if angle < 0:
		angle += 360
		if angle > 180:
			angle = 360 - angle
	elif angle > 180:
		angle = 360 - angle
	if draw:
		cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 3)
		cv2.line(img, (x3, y3), (x2, y2), (255,255,255), 3)
		cv2.circle(img, (x1, y1), 5, (75,0,130), cv2.FILLED)
		cv2.circle(img, (x1, y1), 15, (75,0,130), 2)
		cv2.circle(img, (x2, y2), 5, (75,0,130), cv2.FILLED)
		cv2.circle(img, (x2, y2), 15, (75,0,130), 2)
		cv2.circle(img, (x3, y3), 5, (75,0,130), cv2.FILLED)
		cv2.circle(img, (x3, y3), 15, (75,0,130), 2)
		cv2.putText(img, str(int(angle)), (x2-50, y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
	return angle


def set_percentage_bar_and_text(elbow_angle, knee_angle, shoulder_angle, workout_name_after_smoothening):
	if workout_name_after_smoothening == "pushups":
		success_percentage = np.interp(elbow_angle, (90, 160), (0, 100))
		progress_bar = np.interp(elbow_angle, (90, 160), (380, 30))
	elif workout_name_after_smoothening == "squats":
		success_percentage = np.interp(knee_angle, (90, 160), (0, 100))
		progress_bar = np.interp(knee_angle, (90, 160), (380, 30))
	elif workout_name_after_smoothening == "jumping_jacks":
		success_percentage = np.interp(shoulder_angle, (40, 160), (0, 100))
		progress_bar = np.interp(shoulder_angle, (40, 160), (380, 30))
	else:
		success_percentage = 0
		progress_bar = 380
	return success_percentage, progress_bar


def set_body_angles_from_keypoints(get_angle_fn, img, landmark_list):
	elbow_angle = get_angle_fn(img, landmark_list, 11, 13, 15)
	shoulder_angle = get_angle_fn(img, landmark_list, 13, 11, 23)
	hip_angle = get_angle_fn(img, landmark_list, 11, 23,25)
	elbow_angle_right = get_angle_fn(img, landmark_list, 12, 14, 16)
	shoulder_angle_right = get_angle_fn(img, landmark_list, 14, 12, 24)
	hip_angle_right = get_angle_fn(img, landmark_list, 12, 24,26)
	knee_angle = get_angle_fn(img, landmark_list, 24,26, 28)
	return elbow_angle,shoulder_angle,hip_angle,elbow_angle_right,shoulder_angle_right,hip_angle_right,knee_angle


def check_form(elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, knee_angle, form, workout_name_after_smoothening):
	if workout_name_after_smoothening == "pushups":
		if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160 and elbow_angle_right > 160 and shoulder_angle_right > 40 and hip_angle_right > 160:
			form = 1
	else:
		if knee_angle > 160:
			form = 1
	return form


def display_workout_stats(count, form, feedback, draw_percentage_progress_bar, display_rep_count, show_workout_feedback, show_workout_name_from_model, img, pushup_success_percentage, pushup_progress_bar, workout_name_after_smoothening):
	draw_percentage_progress_bar(form, img, pushup_success_percentage, pushup_progress_bar)
	display_rep_count(count, img)
	show_workout_feedback(feedback, img)
	show_workout_name_from_model(img, workout_name_after_smoothening)
