import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  #0第一個攝影機
mpHands = mp.solutions.hands  #追蹤手部
                      #偵測嚴謹度 0-1 越高越嚴謹        #重新測測的速度 越高越快   預設皆為0.5
hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3) #設定點的顏色  紅色
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5) #設定線的顏色及粗度  綠色
pTime = 0
cTime = 0

while True:
    ret, img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#將圖片轉成RGB
        result = hands.process(imgRGB)

        # print(result.multi_hand_landmarks)   #偵測到手的21個座標
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        if result.multi_hand_landmarks:        #如果偵測到手
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                for i, lm in enumerate(handLms.landmark): #印出21個點的座標 i:點 lm:座標
                    xPos = int(lm.x * imgWidth)           #去掉小數點
                    yPos = int(lm.y * imgHeight)

                    # cv2.putText(img, str(i), (xPos-25, yPos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

                    # if i == 4:
                    #     cv2.circle(img, (xPos, yPos), 20, (166, 56, 56), cv2.FILLED)
                    # print(i, xPos, yPos)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)#設置FPS

        cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break