import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from FitnessTrainerLSTM import FitnessTrainerLSTM
import numpy as np
from collections import deque
import math


# Mapping dictionary to map keypoints from Mediapipe to our Classifier model
lm_dict = {
  0:0 , 1:10, 2:12, 3:14, 4:16, 5:11, 6:13, 7:15, 8:24, 9:26, 10:28, 11:23, 12:25, 13:27, 14:5, 15:2, 16:8, 17:7,
}



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


def get_pose (img, results, draw=True):        
        if results.pose_landmarks:
            if draw:
                mpDraw = mp.solutions.drawing_utils
                mpDraw.draw_landmarks(img,results.pose_landmarks,
                                           mpPose.POSE_CONNECTIONS) 
        return img

def get_position(img, results, height, width, draw=True):
        landmark_list = []
        if results.pose_landmarks:
            for id, landmark in enumerate(results.pose_landmarks.landmark):
                #finding height, width of the image printed
                height, width, c = img.shape
                #Determining the pixels of the landmarks
                landmark_pixel_x, landmark_pixel_y = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append([id, landmark_pixel_x, landmark_pixel_y])
                if draw:
                    cv2.circle(img, (landmark_pixel_x, landmark_pixel_y), 5, (255,0,0), cv2.FILLED)
        return landmark_list    


def get_angle(img, landmark_list, point1, point2, point3, draw=True):   
        #Retrieve landmark coordinates from point identifiers
        x1, y1 = landmark_list[point1][1:]
        x2, y2 = landmark_list[point2][1:]
        x3, y3 = landmark_list[point3][1:]
            
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - 
                             math.atan2(y1-y2, x1-x2))
        
        #Handling angle edge cases: Obtuse and negative angles
        if angle < 0:
            angle += 360
            if angle > 180:
                angle = 360 - angle
        elif angle > 180:
            angle = 360 - angle
            
        if draw:
            #Drawing lines between the three points
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255,255,255), 3)

            #Drawing circles at intersection points of lines
            cv2.circle(img, (x1, y1), 5, (75,0,130), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (75,0,130), 2)
            cv2.circle(img, (x2, y2), 5, (75,0,130), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (75,0,130), 2)
            cv2.circle(img, (x3, y3), 5, (75,0,130), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (75,0,130), 2)
            
            #Show angles between lines
            cv2.putText(img, str(int(angle)), (x2-50, y2+50), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        return angle

    
    
def convert_mediapipe_keypoints_for_model(lm_dict, landmark_list):
    inp_pushup = []
    for index in range(0, 36):
        if index < 18:
            inp_pushup.append(round(landmark_list[lm_dict[index]][1],3))
        else:
            inp_pushup.append(round(landmark_list[lm_dict[index-18]][2],3))
    return inp_pushup



# Setting variables for video feed
def set_video_feed_variables():
    cap = cv2.VideoCapture(0)
    count = 0
    direction = 0
    form = 0
    feedback = "Get into Position! Lets Start the workout!"
    frame_queue = deque(maxlen=250)
    clf = FitnessTrainerLSTM(r'C:\Users\whack\OneDrive\Desktop\ML\FitnessTrainer\models\fitness_trainer.tflite')
    return cap,count,direction,form,feedback,frame_queue,clf


def set_percentage_bar_and_text(elbow_angle, knee_angle, shoulder_angle, workout_name_after_smoothening):
    if workout_name_after_smoothening == "pushups":
        success_percentage = np.interp(elbow_angle, (90, 160), (0, 100))
        progress_bar = np.interp(elbow_angle, (90, 160), (380, 30))
    elif workout_name_after_smoothening == "squats":
        success_percentage = np.interp(knee_angle, (90, 160), (0, 100))
        progress_bar = np.interp(knee_angle, (90, 160), (380, 30))
    elif workout_name_after_smoothening == "jumping jacks":
        success_percentage = np.interp(shoulder_angle, (40, 160), (0, 100)) 
        progress_bar = np.interp(shoulder_angle, (40, 160), (380, 30))
    else:
        success_percentage = 0
        progress_bar = 380

    return success_percentage, progress_bar


def set_body_angles_from_keypoints(get_angle, img, landmark_list):
    elbow_angle = get_angle(img, landmark_list, 11, 13, 15)
    shoulder_angle = get_angle(img, landmark_list, 13, 11, 23)
    hip_angle = get_angle(img, landmark_list, 11, 23,25)
    elbow_angle_right = get_angle(img, landmark_list, 12, 14, 16)
    shoulder_angle_right = get_angle(img, landmark_list, 14, 12, 24)
    hip_angle_right = get_angle(img, landmark_list, 12, 24,26)
    knee_angle = get_angle(img, landmark_list, 24,26, 28)
    return elbow_angle,shoulder_angle,hip_angle,elbow_angle_right,shoulder_angle_right,hip_angle_right,knee_angle

def set_smoothened_workout_name(lm_dict, convert_mediapipe_keypoints_for_model, frame_queue, clf, landmark_list):
    inp_pushup = convert_mediapipe_keypoints_for_model(lm_dict, landmark_list)
    workout_name = clf.predict(inp_pushup)
    frame_queue.append(workout_name)
    workout_name_after_smoothening = max(set(frame_queue), key=frame_queue.count)
    return "Workout Name: " + workout_name_after_smoothening

def draw_percentage_progress_bar(form, img, pushup_success_percentage, pushup_progress_bar):
    xd, yd, wd, hd = 10, 175, 50, 200
    if form == 1:
        cv2.rectangle(img, (xd,30), (xd+wd, yd+hd), (0, 255, 0), 3)
        cv2.rectangle(img, (xd, int(pushup_progress_bar)), (xd+wd, yd+hd), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(pushup_success_percentage)}%', (xd, yd+hd+50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 2)

def display_rep_count(count, img):
    xc, yc = 85, 100
    cv2.putText(img, "Reps: " + str(int(count)), (xc, yc), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 2)

def show_workout_feedback(feedback, img):    
    xf, yf = 85, 70
    cv2.putText(img, feedback, (xf, yf), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0,0,0), 2)

def show_workout_name_from_model(img, workout_name_after_smoothening):
    xw, yw = 85, 40
    cv2.putText(img, workout_name_after_smoothening, (xw,yw), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0,0,0), 2)

def check_form(elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, knee_angle, form, workout_name_after_smoothening):
    if workout_name_after_smoothening == "pushups":
        if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160 and elbow_angle_right > 160 and shoulder_angle_right > 40 and hip_angle_right > 160:
            form = 1
    # For now, else impleements squats condition        
    else:
        if knee_angle > 160:
            form = 1
    return form

def display_workout_stats(count, form, feedback, draw_percentage_progress_bar, display_rep_count, show_workout_feedback, show_workout_name_from_model, img, pushup_success_percentage, pushup_progress_bar, workout_name_after_smoothening):
    #Draw the pushup progress bar
    draw_percentage_progress_bar(form, img, pushup_success_percentage, pushup_progress_bar)

    #Show the rep count
    display_rep_count(count, img)
        
    #Show the pushup feedback 
    show_workout_feedback(feedback, img)
        
    #Show workout name
    show_workout_name_from_model(img, workout_name_after_smoothening)


def main():
    mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, detectionCon, trackCon, mpPose = set_pose_parameters()
    pose = mpPose.Pose(mode, complexity, smooth_landmarks,
                                enable_segmentation, smooth_segmentation,
                                detectionCon, trackCon)

    # Setting video feed variables
    cap, count, direction, form, feedback, frame_queue, clf = set_video_feed_variables()

    # Get user input for the desired exercise
    exercise_choice = int(input("---Choose your exercise number---\n1. Bicep Curls\n2. Squats\n3. Jumping Jacks\n4. Shoulder Press\n5. Pushups\nChoose: "))
    
    # Validate user input
    exercise_list = ["bicep_curls", "squats", "jumping_jacks", "shoulder_press", "pushups"]
    if exercise_choice > len(exercise_list):
        print("Invalid Choice!")
        return
    
    # Set the fullscreen mode before the loop, using the same window name as in imshow
    cv2.namedWindow('TRAINIFY!', cv2.WINDOW_NORMAL)  # Set to normal first to avoid conflicts
    cv2.setWindowProperty('TRAINIFY!', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    exercise_choice = exercise_list[exercise_choice - 1]
    
    # initial values
    jumping_jack_stage = "down"
    squat_stage = "up"
    bicep_stage = "down"
    shoulder_press_stage = "down"
    pushup_stage = "up"

    # Start video feed and run workout
    while cap.isOpened():
        # Getting image from camera
        ret, img = cap.read()
        # Getting video dimensions
        width = cap.get(3)
        height = cap.get(4)

        # Convert from BGR (used by cv2) to RGB (used by Mediapipe)
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Get pose and draw landmarks
        img = get_pose(img, results, False)

        # Get landmark list from mediapipe
        landmark_list = get_position(img, results, height, width, False)

        # If landmarks exist, get the relevant workout body angles
        if len(landmark_list) != 0:
            elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, knee_angle = set_body_angles_from_keypoints(get_angle, img, landmark_list)

            # Use the chosen exercise directly instead of predictions
            workout_name_after_smoothening = exercise_choice

            pushup_success_percentage, pushup_progress_bar = set_percentage_bar_and_text(elbow_angle, knee_angle, shoulder_angle, workout_name_after_smoothening)

            # Is the form correct at the start?
            form = check_form(elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, knee_angle, form, workout_name_after_smoothening)

            # THIS IS BICEP CURLS
            if workout_name_after_smoothening.strip() == "bicep_curls":
                print("Inside Bicep Curls")
                print("Elbow Angle: ", elbow_angle)

                if elbow_angle > 170:
                    print("--------DOWN STAGE--------")
                    if bicep_stage == "up":
                        feedback = "Perfect Rep!"
                        count += 1
                        bicep_stage = "down"
                        
                    feedback = "Slowly perform the bicep curl"

                if elbow_angle < 60 and bicep_stage == "down":
                    print("-------UP STAGE-----")
                    feedback = "Good curl, now go back down"
                    bicep_stage = "up"

            # THIS IS SQUATS
            if workout_name_after_smoothening.strip() == "squats":
                print("Inside Squats")
                print("Hip Angle: ", hip_angle, "\tKnee Angle: ", knee_angle)

                # Check if the person is in the 'down' position (squatting phase)
                if hip_angle < 90 and knee_angle < 110:
                    print('----------DOWN STAGE------------')
                    feedback = "HOLD for few seconds!"
                    squat_stage = "down"
                    print("Currently in squat down position!")

                # Check if the person has returned to the 'up' position (standing phase)
                elif hip_angle > 160 and knee_angle > 160:
                    print("----------------Standing Up!------------------")
                    if squat_stage == 'down':
                        count += 1
                        feedback = "Perfect Squat"
                        squat_stage = 'up'
                    else:
                        feedback = "Squat Down"
                
                print(count)

            # THIS IS JUMPING JACKS
            if workout_name_after_smoothening == "jumping_jacks":
                print("Shoulder Angle: ", shoulder_angle, "\hip Angle: ", hip_angle)
                
                up_stage = 0
                # Perform jumping jacks check (e.g., knees should bend during the down phase and arms move during the up phase)
                if shoulder_angle < 90 and hip_angle > 163:
                    if jumping_jack_stage == "up":
                        feedback = "Perfectly Done!"
                        jumping_jack_stage = "down"
                    else:
                        # Person is in 'down' position of jumping jack
                        feedback = "Jump down detected, now jump up!"

                elif shoulder_angle > 120 and hip_angle < 158 and jumping_jack_stage == 'down':
                    # Person is in 'up' position of jumping jack
                    if up_stage == 0:
                        count += 1
                        up_stage = 1
                    feedback = "Now go back down!"
                    jumping_jack_stage = "up"
            

            # THIS IS SHOULDER PRESS
            if workout_name_after_smoothening == "shoulder_press":
                print("Inside Shoulder Press")
                # Default standing position
                if shoulder_angle < 40:
                    feedback = "Bring your elbow to your Shoulder\n Fist facing Up!"
                
                # 90 Degree angle
                if shoulder_angle > 60 and shoulder_angle < 110:
                    feedback = "Starting position!\nPush your arm up!"

                    if shoulder_press_stage == "up":
                        count += 1
                        feedback = "Perfect Rep!"
                        shoulder_press_stage = "mid"

                if shoulder_angle > 170 and elbow_angle > 170:
                    feedback = "Good Press, bring arm back\nto Starting Point!"
                    shoulder_press_stage = "up"

            
            # THIS IS PUSHUPS
            if workout_name_after_smoothening == "pushups":
                
                # Initial Stage
                if shoulder_angle < 50 and shoulder_angle > 20 and hip_angle > 165 and elbow_angle > 120:
                    feedback = "Starting Position, bend elbow\nGo Down!"

                    if pushup_stage == "down":
                        count += 1
                        feedback = "Perfect Rep!"
                        pushup_stage = "up"

                # Down Stage
                if shoulder_angle < 20 and hip_angle > 165 and elbow_angle < 90:
                    feedback = "Good, now come back up"
                    pushup_stage = "down"
                
                # Hip Problems
                if hip_angle < 155:
                    feedback = "Straigthen Hips!"

            # Display workout stats        
            display_workout_stats(count, form, feedback, draw_percentage_progress_bar, display_rep_count, show_workout_feedback, show_workout_name_from_model, img, pushup_success_percentage, pushup_progress_bar, workout_name_after_smoothening)
            
        # Transparent Overlay
        overlay = img.copy()
        x, y, w, h = 75, 10, 500, 150
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), -1)      
        alpha = 0.8  # Transparency factor.
        # Following line overlays transparent rectangle over the image
        image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)          
            
        cv2.imshow('TRAINIFY!', image_new)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
