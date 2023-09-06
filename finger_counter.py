#import the required packages
import mediapipe 
import cv2

#hand detection
medhands = mediapipe.solutions.hands
draw = mediapipe.solutions.drawing_utils

hand = medhands.Hands(max_num_hands=1)

video = cv2.VideoCapture(0)

while True:
    sucess,img = video.read()
    img = cv2.flip(img,1)
    imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    res = hand.process(imgrgb)
    tipids = [4,8,12,16,20]
    lmlist = []


    cv2.rectangle(img,(20,350),(90,440),(123,23,233),cv2.FILLED)
    cv2.rectangle(img,(20,350),(90,440),(123,123,123),5)

    if res.multi_hand_landmarks:
        for handlms in res.multi_hand_landmarks:
            for id,lm in enumerate(handlms.landmark):
                #print(id,lm)
                cx = lm.x
                cy = lm.y
                lmlist.append([id,cx,cy])
                if len(lmlist)!=0 and len(lmlist) == 21:
                    fingerlist=[]
                    #thumb
                    if lmlist[12][1]>lmlist[20][1]: #left hand
                        if lmlist[4][1]<lmlist[3][1]: #finger closed
                            fingerlist.append(0)
                        else:
                            fingerlist.append(1)
                    else:
                        if lmlist[4][1]>lmlist[3][1]:
                            fingerlist.append(0)
                        else:
                            fingerlist.append(1)

                    #other fingers
                    for i in range(1,5):
                        if lmlist[tipids[i]][2]<lmlist[tipids[i]-2][2]: #to fetch cy value
                            fingerlist.append(1)#finger open
                        else:
                            fingerlist.append(0) #closed
        #print(fingerlist)
        if len(fingerlist)!=0:
            fingercount = fingerlist.count(1)
        cv2.putText(img,str(fingercount),org=(35,436),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=3,color=(255,0,0),thickness=4)
        
    
        #print(lmlist)
        draw.draw_landmarks(img,handlms,medhands.HAND_CONNECTIONS,draw.DrawingSpec(color = (0,0,255),thickness=2,circle_radius=3),draw.DrawingSpec(color=(255,0,0),thickness=2))
    

    cv2.imshow("HAND",img)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

