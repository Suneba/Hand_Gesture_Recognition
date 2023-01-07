import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
text_speech = pyttsx3.init()
text_speech.setProperty('rate',100)
import threading

#HOPE AND GH_PQ_CO
#L0(HOPE) TO ASEMNT_KVURWFB
#
labelfile= "Model/labels.txt"


lol1 = 123
lol2 = 456
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

classifier = Classifier("Model/GHPQ_ASEMNTIJY_KVURWFB.h5",labelfile)
labels = ["GHPQCO","ASEMNTJY","KVURWF"]
####################################################################
classifier_LAYER1_GHPQCO = Classifier("Model/GH_PQ_CO.h5",labelfile)
layer1_GHPQCO = ["GH","PQ","CO"]

classifier_LAYER0_LAYER1_ASE_MNT_IJ_Y = Classifier("Model/ASE_MNT_IJ_Y.h5",labelfile)
LAYER0_LAYER1_ASE_MNT_IJ_Y = ["ASE","MNT","IJ","Y"]
#
classifier_LAYER0_LAYER1_KVUR_WFB = Classifier("Model/KVUR_WFB.h5",labelfile)
LAYER0_LAYER1_KVUR_WFB = ["KVUR","WFB"]
###################################################################
classifier_GH = Classifier("Model/G_H.h5",labelfile)
LABEL_GH = ["G","H"]
classifier_PQ = Classifier("Model/P_Q.h5",labelfile)
LABEL_PQ = ["P","Q"]
classifier_CO = Classifier("Model/C_O.h5",labelfile)
LABEL_CO = ["C","O"]
#############################################################
classifier_ASE = Classifier("Model/A_S_E.h5",labelfile)
LABEL_ASE = ["A","S","E"]
classifier_MNT = Classifier("Model/M_N_T.h5",labelfile)
LABEL_MNT = ["M","N","T"]
classifier_IJY = Classifier("Model/I_J_Y.h5",labelfile)
LABEL_IJY = ["I","J","Y"]
#############################################################
classifier_KV_UR = Classifier("Model/KV_UR.h5",labelfile)
classifier_K_V =  Classifier("Model/K_V.h5",labelfile)
classifier_U_R =  Classifier("Model/U_R.h5",labelfile)
LABEL_KV_UR = ["KV","UR"]
LABEL_KV = ["K","V"]
LABEL_UR = ["U","R"]
#K_V
#U_R
classifier_W_FB = Classifier("Model/W_FB.h5",labelfile)
LABEL_W_FB = ["W","FB"]
classifier_FB = Classifier("Model/F_B.h5",labelfile)
LABEL_FB = ["F","B"]


# classifier_layer1_AE("path to ae file")


offset = 20
imgSize = 300

folder = "Data/C"
counter = 0

def say(textcurr, textprev):

        text_speech.say(str(textcurr))
        text_speech.runAndWait()

def say_letter_typed(letter):
    text_speech.say(str(letter))
    text_speech.runAndWait()



realcounter = 0
textcurr = "null"
textprev = "null"
text_stack =[]




while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            # print(prediction, index)
            cv2.putText(imgOutput, f"{index}", (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)


            if index == 0:#if HOPE-> GHPQCO
                prediction_layer1, index_layer1 = classifier_LAYER1_GHPQCO.getPrediction(imgWhite, draw=False)
                cv2.putText(imgOutput, f"0", (x, y - 26), cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)

                if index_layer1 ==0:#GPQCO->GH
                    prediction_layer1, index_layer1_gh = classifier_GH.getPrediction(imgWhite, draw=False)
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, f"0,{LABEL_GH[index_layer1_gh]}", (x, y - 26), cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                    textcurr = str(LABEL_GH[index_layer1_gh])


                if index_layer1 ==1:#GHPQCO->PQ
                    prediction_layer1, index_layer1_pq = classifier_PQ.getPrediction(imgWhite, draw=False)
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, f"0,{LABEL_PQ[index_layer1_pq]}", (x, y - 26), cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                    textcurr = str(LABEL_PQ[index_layer1_pq])

                if index_layer1 ==2:#GPQCO->CO
                    prediction_layer1, index_layer1_co = classifier_CO.getPrediction(imgWhite, draw=False)
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, f"0,{LABEL_CO[index_layer1_co]}", (x, y - 26), cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                    textcurr = str(LABEL_CO[index_layer1_co])

            #
            #
            elif index == 1:#if HOPE->ASEMNTIJY
                prediction_layer1, index_layer1_ase_mnt_ij_y = classifier_LAYER0_LAYER1_ASE_MNT_IJ_Y.getPrediction(imgWhite, draw=False)
                cv2.putText(imgOutput, f"1", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)

                if index_layer1_ase_mnt_ij_y == 0:#ase
                    prediction_layer1, index_layer1_ase = classifier_ASE.getPrediction(imgWhite, draw=False)
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, f"1,{LABEL_ASE[index_layer1_ase]}", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                    textcurr = str(LABEL_ASE[index_layer1_ase])

                if index_layer1_ase_mnt_ij_y == 1:#mnt
                    prediction_layer1, index_layer1_mnt = classifier_MNT.getPrediction(imgWhite, draw=False)
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, f"1,{LABEL_MNT[index_layer1_mnt]}", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                    textcurr = str(LABEL_MNT[index_layer1_mnt])


                if index_layer1_ase_mnt_ij_y == 2:#ijy>IJ
                    prediction_layer1, index_layer1_ijy = classifier_IJY.getPrediction(imgWhite, draw=False)
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, f"1,{LABEL_IJY[index_layer1_ijy]}", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                    textcurr = str(LABEL_IJY[index_layer1_ijy])

                if index_layer1_ase_mnt_ij_y == 3:#ijy>Y
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, f"1,Y", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                    textcurr = str("y")

            else:#if HOPE->KVURWFB
                prediction_layer1, index_layer1_kvur_wfb = classifier_LAYER0_LAYER1_KVUR_WFB.getPrediction(imgWhite, draw=False)
                cv2.putText(imgOutput, f"2", (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

                if index_layer1_kvur_wfb == 0:#kv_ur
                    prediction_layer1, index_layer1_kvur_wfb = classifier_KV_UR.getPrediction(imgWhite, draw=False)
                    cv2.putText(imgOutput, f"2", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)

                    if index_layer1_kvur_wfb == 0:  # kv
                        prediction_layer1, index_layer1_kvur_wfb_kv = classifier_K_V.getPrediction(imgWhite, draw=False)
                        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                        cv2.putText(imgOutput, f"2,{LABEL_KV[index_layer1_kvur_wfb_kv]}", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                        textcurr = str(LABEL_KV[index_layer1_kvur_wfb_kv])
                    #
                    if index_layer1_kvur_wfb == 1:#ur
                        prediction_layer1, index_layer1_kvur_wfb_ur = classifier_U_R.getPrediction(imgWhite, draw=False)
                        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                        cv2.putText(imgOutput, f"2,{LABEL_UR[index_layer1_kvur_wfb_ur]}", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                        textcurr = str(LABEL_UR[index_layer1_kvur_wfb_ur])


                else:#wfb
                    prediction_layer1, index_layer1_kvur_wfb = classifier_W_FB.getPrediction(imgWhite, draw=False)
                    cv2.putText(imgOutput, f"2", (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

                    if index_layer1_kvur_wfb == 0: #print_w
                        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                        cv2.putText(imgOutput, f"2,w", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                        textcurr = str("w")
                    else:#search for f_b
                        prediction_layer1, index_layer1_kvur_wfb_fb = classifier_FB.getPrediction(imgWhite, draw=False)
                        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                        cv2.putText(imgOutput, f"2,{LABEL_FB[index_layer1_kvur_wfb_fb]}", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                        textcurr = str(LABEL_FB[index_layer1_kvur_wfb_fb])







        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            cv2.putText(imgOutput, f"{index}", (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)


            if index == 0:#if HOPE-> GHPQCO
                prediction_layer1, index_layer1 = classifier_LAYER1_GHPQCO.getPrediction(imgWhite, draw=False)
                cv2.putText(imgOutput, f"0", (x, y - 26), cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)

                if index_layer1 ==0:#GPQCO->GH
                    prediction_layer1, index_layer1_gh = classifier_GH.getPrediction(imgWhite, draw=False)
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, f"0,{LABEL_GH[index_layer1_gh]}", (x, y - 26), cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                    textcurr = str(LABEL_GH[index_layer1_gh])


                if index_layer1 ==1:#GHPQCO->PQ
                    prediction_layer1, index_layer1_pq = classifier_PQ.getPrediction(imgWhite, draw=False)
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, f"0,{LABEL_PQ[index_layer1_pq]}", (x, y - 26), cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                    textcurr = str(LABEL_PQ[index_layer1_pq])

                if index_layer1 ==2:#GPQCO->CO
                    prediction_layer1, index_layer1_co = classifier_CO.getPrediction(imgWhite, draw=False)
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, f"0,{LABEL_CO[index_layer1_co]}", (x, y - 26), cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                    textcurr = str(LABEL_CO[index_layer1_co])

            #
            #
            elif index == 1:#if HOPE->ASEMNTIJY
                prediction_layer1, index_layer1_ase_mnt_ij_y = classifier_LAYER0_LAYER1_ASE_MNT_IJ_Y.getPrediction(imgWhite, draw=False)
                cv2.putText(imgOutput, f"1", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)

                if index_layer1_ase_mnt_ij_y == 0:#ase
                    prediction_layer1, index_layer1_ase = classifier_ASE.getPrediction(imgWhite, draw=False)
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, f"1,{LABEL_ASE[index_layer1_ase]}", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                    textcurr = str(LABEL_ASE[index_layer1_ase])

                if index_layer1_ase_mnt_ij_y == 1:#mnt
                    prediction_layer1, index_layer1_mnt = classifier_MNT.getPrediction(imgWhite, draw=False)
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, f"1,{LABEL_MNT[index_layer1_mnt]}", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                    textcurr = str(LABEL_MNT[index_layer1_mnt])


                if index_layer1_ase_mnt_ij_y == 2:#ijy>IJ
                    prediction_layer1, index_layer1_ijy = classifier_IJY.getPrediction(imgWhite, draw=False)
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, f"1,{LABEL_IJY[index_layer1_ijy]}", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                    textcurr = str(LABEL_IJY[index_layer1_ijy])

                if index_layer1_ase_mnt_ij_y == 3:#ijy>Y
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, f"1,Y", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                    textcurr = str("y")

            else:#if HOPE->KVURWFB
                prediction_layer1, index_layer1_kvur_wfb = classifier_LAYER0_LAYER1_KVUR_WFB.getPrediction(imgWhite, draw=False)
                cv2.putText(imgOutput, f"2", (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

                if index_layer1_kvur_wfb == 0:#kv_ur
                    prediction_layer1, index_layer1_kvur_wfb = classifier_KV_UR.getPrediction(imgWhite, draw=False)
                    cv2.putText(imgOutput, f"2", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)

                    if index_layer1_kvur_wfb == 0:  # kv
                        prediction_layer1, index_layer1_kvur_wfb_kv = classifier_K_V.getPrediction(imgWhite, draw=False)
                        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                        cv2.putText(imgOutput, f"2,{LABEL_KV[index_layer1_kvur_wfb_kv]}", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                        textcurr = str(LABEL_KV[index_layer1_kvur_wfb_kv])
                    #
                    if index_layer1_kvur_wfb == 1:#ur
                        prediction_layer1, index_layer1_kvur_wfb_ur = classifier_U_R.getPrediction(imgWhite, draw=False)
                        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                        cv2.putText(imgOutput, f"2,{LABEL_UR[index_layer1_kvur_wfb_ur]}", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                        textcurr = str(LABEL_UR[index_layer1_kvur_wfb_ur])


                else:#wfb
                    prediction_layer1, index_layer1_kvur_wfb = classifier_W_FB.getPrediction(imgWhite, draw=False)
                    cv2.putText(imgOutput, f"2", (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

                    if index_layer1_kvur_wfb == 0: #print_w
                        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                        cv2.putText(imgOutput, f"2,w", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                        textcurr = str("w")
                    else:#search for f_b
                        prediction_layer1, index_layer1_kvur_wfb_fb = classifier_FB.getPrediction(imgWhite, draw=False)
                        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),(255, 0, 255), cv2.FILLED)
                        cv2.putText(imgOutput, f"2,{LABEL_FB[index_layer1_kvur_wfb_fb]}", (x, y - 26),cv2.FONT_HERSHEY_COMPLEX,1.7, (255, 255, 255), 2)
                        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                        textcurr = str(LABEL_FB[index_layer1_kvur_wfb_fb])

        # cv2.imshow("ImageCrop", imgCrop)
        # cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

    if key == ord("d"):
        threading.Thread(target=say_letter_typed, args=("d")).start()
    if key == ord("z"):
        threading.Thread(target=say_letter_typed, args=("z")).start()
    if key == ord("x"):
        threading.Thread(target=say_letter_typed, args=("x")).start()
    if key == ord("l"):
        threading.Thread(target=say_letter_typed, args=("l")).start()


    if textcurr != "null" and textprev != textcurr:
        threading.Thread(target=say, args=(textcurr, textprev)).start()
        textprev = textcurr
        textcurr = "null"







