import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips =[8, 12, 16, 20]
thumb_tip= 4

while True:
    ret,img = cap.read()
    img = cv2.flip(img, 1)
    h,w,c = img.shape
    results = hands.process(img)

    finger_fold_status = []
    thumb_up = False
    thumb_down = False

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            #accessing the landmarks by their position
            lm_list=[]
            for id ,lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            fingertips = []
            for tip_id in finger_tips:
                tip_x, tip_y = int(lm_list[tip_id].x * w), int(lm_list[tip_id].y * h)
                fingertips.append((tip_x, tip_y))
                cv2.circle(img, (tip_x, tip_y), 10, (255, 0, 0), -1)

                # check if finger is folded
                finger_folded = lm_list[tip_id].x > lm_list[tip_id - 2].x
                finger_fold_status.append(finger_folded)

            thumb_tip_x, thumb_tip_y = int(lm_list[thumb_tip].x * w), int(lm_list[thumb_tip].y * h)
            thumb_up = thumb_tip_y < fingertips[0][1]
            thumb_down = thumb_tip_y > fingertips[0][1]

        if all(finger_fold_status):
            if thumb_up:
                print("Thumbs up!")
                cv2.putText(img, "LIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif thumb_down:
                print("Thumbs down!")
                cv2.putText(img, "DISLIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        mp_draw.draw_landmarks(img, hand_landmark,
        mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0,0,255),2,2),
        mp_draw.DrawingSpec((0,255,0),4,2))

    cv2.imshow("hand tracking", img)
    cv2.waitKey(1)
