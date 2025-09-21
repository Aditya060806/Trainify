import av
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import streamlit as st
from pose_utils import (
	set_pose_parameters,
	get_angle,
	set_percentage_bar_and_text,
	set_body_angles_from_keypoints,
	check_form,
	display_workout_stats,
)

# App title
st.set_page_config(page_title="Trainify - AI Fitness Trainer", layout="wide")
st.title("Trainify - AI Fitness Trainer")
st.markdown("Choose an exercise and start the webcam to track your reps and get feedback.")

# Sidebar controls
exercise_names = ["bicep_curls", "squats", "jumping_jacks", "shoulder_press", "pushups"]
exercise_choice = st.sidebar.selectbox("Exercise", exercise_names, index=4)
show_landmarks = st.sidebar.checkbox("Show landmarks", value=False)

# State for counters
if "rep_count" not in st.session_state:
	st.session_state.rep_count = 0
if "feedback" not in st.session_state:
	st.session_state.feedback = "Get into Position! Lets Start the workout!"
if "form_ok" not in st.session_state:
	st.session_state.form_ok = 0

# Pose setup (created once)
mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, detectionCon, trackCon, mpPose = set_pose_parameters()
pose = mpPose.Pose(mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, detectionCon, trackCon)

drawing_utils = mp.solutions.drawing_utils
pose_connections = mp.solutions.pose.POSE_CONNECTIONS

class ExerciseTransformer(VideoTransformerBase):
	def __init__(self):
		self.count = 0
		self.feedback = "Get into Position! Lets Start the workout!"
		self.form = 0
		self.direction = 0
		self.jumping_jack_stage = "down"
		self.squat_stage = "up"
		self.bicep_stage = "down"
		self.shoulder_press_stage = "down"
		self.pushup_stage = "up"

	def transform(self, frame: av.VideoFrame) -> np.ndarray:
		img = frame.to_ndarray(format="bgr24")
		results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

		# Draw pose landmarks
		if show_landmarks and results.pose_landmarks:
			drawing_utils.draw_landmarks(img, results.pose_landmarks, pose_connections)

		if not results.pose_landmarks:
			return img

		height, width, _ = img.shape
		landmark_list = []
		for lid, landmark in enumerate(results.pose_landmarks.landmark):
			landmark_pixel_x, landmark_pixel_y = int(landmark.x * width), int(landmark.y * height)
			landmark_list.append([lid, landmark_pixel_x, landmark_pixel_y])

		(
			elbow_angle,
			shoulder_angle,
			hip_angle,
			elbow_angle_right,
			shoulder_angle_right,
			hip_angle_right,
			knee_angle,
		) = set_body_angles_from_keypoints(get_angle, img, landmark_list)

		workout_name_after_smoothening = exercise_choice

		pushup_success_percentage, pushup_progress_bar = set_percentage_bar_and_text(
			elbow_angle, knee_angle, shoulder_angle, workout_name_after_smoothening
		)

		self.form = check_form(
			elbow_angle,
			shoulder_angle,
			hip_angle,
			elbow_angle_right,
			shoulder_angle_right,
			hip_angle_right,
			knee_angle,
			self.form,
			workout_name_after_smoothening,
		)

		if workout_name_after_smoothening == "bicep_curls":
			if elbow_angle > 170:
				if self.bicep_stage == "up":
					self.feedback = "Perfect Rep!"
					self.count += 1
					self.bicep_stage = "down"
				self.feedback = "Slowly perform the bicep curl"
			if elbow_angle < 60 and self.bicep_stage == "down":
				self.feedback = "Good curl, now go back down"
				self.bicep_stage = "up"

		elif workout_name_after_smoothening == "squats":
			if hip_angle < 90 and knee_angle < 110:
				self.feedback = "HOLD for few seconds!"
				self.squat_stage = "down"
			elif hip_angle > 160 and knee_angle > 160:
				if self.squat_stage == "down":
					self.count += 1
					self.feedback = "Perfect Squat"
					self.squat_stage = "up"
				else:
					self.feedback = "Squat Down"

		elif workout_name_after_smoothening == "jumping_jacks":
			up_stage = 0
			if shoulder_angle < 90 and hip_angle > 163:
				if self.jumping_jack_stage == "up":
					self.feedback = "Perfectly Done!"
					self.jumping_jack_stage = "down"
				else:
					self.feedback = "Jump down detected, now jump up!"
			elif shoulder_angle > 120 and hip_angle < 158 and self.jumping_jack_stage == "down":
				if up_stage == 0:
					self.count += 1
					up_stage = 1
				self.feedback = "Now go back down!"
				self.jumping_jack_stage = "up"

		elif workout_name_after_smoothening == "shoulder_press":
			if shoulder_angle < 40:
				self.feedback = "Bring elbow to shoulder, fist up!"
			if shoulder_angle > 60 and shoulder_angle < 110:
				self.feedback = "Starting position! Push up!"
				if self.shoulder_press_stage == "up":
					self.count += 1
					self.feedback = "Perfect Rep!"
					self.shoulder_press_stage = "mid"
			if shoulder_angle > 170 and elbow_angle > 170:
				self.feedback = "Good press, return to start!"
				self.shoulder_press_stage = "up"

		elif workout_name_after_smoothening == "pushups":
			if shoulder_angle < 50 and shoulder_angle > 20 and hip_angle > 165 and elbow_angle > 120:
				self.feedback = "Starting Position, bend elbow — Go Down!"
				if self.pushup_stage == "down":
					self.count += 1
					self.feedback = "Perfect Rep!"
					self.pushup_stage = "up"
			if shoulder_angle < 20 and hip_angle > 165 and elbow_angle < 90:
				self.feedback = "Good, now come back up"
				self.pushup_stage = "down"
			if hip_angle < 155:
				self.feedback = "Straighten Hips!"

		overlay = img.copy()
		x, y, w, h = 75, 10, 500, 150
		cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), -1)
		alpha = 0.8
		image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

		display_workout_stats(
			self.count,
			self.form,
			self.feedback,
			lambda form,img,pct,bar: _draw_bar(form, image_new, pct, bar),
			lambda count,img: _draw_reps(count, image_new),
			lambda feedback,img: _draw_feedback(feedback, image_new),
			lambda img,name: _draw_name(image_new, f"Workout Name: {workout_name_after_smoothening}"),
			image_new,
			pushup_success_percentage,
			pushup_progress_bar,
			workout_name_after_smoothening,
		)

		st.session_state.rep_count = int(self.count)
		st.session_state.feedback = self.feedback
		st.session_state.form_ok = self.form

		return image_new


def _draw_bar(form, img, pct, bar):
	xd, yd, wd, hd = 10, 175, 50, 200
	if form == 1:
		cv2.rectangle(img, (xd,30), (xd+wd, yd+hd), (0, 255, 0), 3)
		cv2.rectangle(img, (xd, int(bar)), (xd+wd, yd+hd), (0, 255, 0), cv2.FILLED)
		cv2.putText(img, f'{int(pct)}%', (xd, yd+hd+50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)


def _draw_reps(count, img):
	xc, yc = 85, 100
	cv2.putText(img, f"Reps: {int(count)}", (xc, yc), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)


def _draw_feedback(feedback, img):
	xf, yf = 85, 70
	cv2.putText(img, feedback, (xf, yf), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)


def _draw_name(img, name):
	xw, yw = 85, 40
	cv2.putText(img, name, (xw,yw), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)


webrtc_ctx = webrtc_streamer(
	key="trainify",
	mode=WebRtcMode.SENDRECV,
	video_transformer_factory=ExerciseTransformer,
	media_stream_constraints={"video": True, "audio": False},
)

col1, col2, col3 = st.columns(3)
with col1:
	st.metric("Reps", st.session_state.rep_count)
with col2:
	st.metric("Form OK", "Yes" if st.session_state.form_ok == 1 else "No")
with col3:
	st.metric("Feedback", st.session_state.feedback)
