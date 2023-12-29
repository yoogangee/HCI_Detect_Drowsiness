import multiprocessing

import cv2
import alphashape
import numpy as np
from scipy.signal import welch, butter, lfilter

import mediapipe as md
import model
import time
from cvzone.FaceMeshModule import FaceMeshDetector
import pygame


def play_sound(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # 음악 재생이 끝날 때까지 대기


def blink(detector, frame):
    # success, img = cap.read()
    frame, faces = detector.findFaceMesh(frame, draw=False)

    if faces:
        face = faces[0]
        # 눈의 중요한 랜드마크
        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lenghtVer, _ = detector.findDistance(leftUp, leftDown)
        lenghtHor, _ = detector.findDistance(leftLeft, leftRight)

        # 눈 깜빡임 비율 계산
        ratio = int((lenghtVer / lenghtHor) * 100)

        # 눈 깜빡임 감지
        if ratio < 35:
            # blink_status = "Blinking"
            close = True
        else:
            # blink_status = "Not Blinking"
            close = False
        return close


def get_hr(y, sr=30, min=30, max=180):
    p, q = welch(y, sr, nfft=1e5 / sr, nperseg=np.min((len(y) - 1, 256)))
    h = p[(p > min / 60) & (p < max / 60)][np.argmax(q[(p > min / 60) & (p < max / 60)])] * 60
    h = h / 1.2
    return h


def frame_collector(collector_pipe, duration=11, fps=30):
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=1)
    with md.solutions.face_mesh.FaceMesh(max_num_faces=1) as fm:
        box = box_ = None
        initial_sent = False
        f = []
        closed_frame = 0
        total_frame = 0

        message = "Measuring heart rate..."
        close = 0
        perclos = 0
        last_sent_time = time.time()
        while True:

            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, message, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

            h, w, c = frame.shape
            if not ret:
                break

            fr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # rppg
            if w > h:
                frame_ = fr[:, round((w - h) / 2):round((w - h) / 2) + h]
            else:
                frame_ = fr
                w = h
                # FaceMesh 처리 및 큐에 전송
            landmarks = fm.process(frame_).multi_face_landmarks
            if landmarks:
                # rppg 모델 전처리
                landmark = np.array([(i.x * h / w + round((w - h) / 2) / w, i.y) for i in landmarks[0].landmark])
                shape = alphashape.alphashape(landmark, 0)
                if box is None:
                    box = np.array(shape.bounds).reshape(2, 2)
                else:
                    w = 1 / (1 + np.exp(-20 * np.linalg.norm(np.array(shape.bounds).reshape(2, 2) - box) / np.multiply(
                        *np.abs(box[0] - box[1])))) * 2 - 1
                    box = np.array(shape.bounds).reshape(2, 2) * w + box * (1 - w)
                if box_ is None:
                    box_ = np.clip(np.round(box * fr.shape[1::-1]).astype(int).T, a_min=0, a_max=None)
                elif np.linalg.norm(np.round(box * fr.shape[1::-1]).astype(int).T - box_) > fr.size / 10 ** 5:
                    box_ = np.clip(np.round(box * fr.shape[1::-1]).astype(int).T, a_min=0, a_max=None)
            else:
                landmark = np.full((468, 2), -1)

            try:
                _ = cv2.resize(fr[slice(*box_[1]), slice(*box_[0])], (8, 8), interpolation=cv2.INTER_AREA)
            except TypeError as e:
                print(e, box_)
            else:
                f.append(_)
                total_frame += 1

            if blink(detector, fr):
                closed_frame += 1
                cv2.putText(frame, "Closed", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)


            if len(f) == 450 and not initial_sent:
                perclos = closed_frame / total_frame
                data = f, perclos
                collector_pipe.send(data)
                print('전송완료', len(f), closed_frame)

                status, hr = collector_pipe.recv()
                message = f"Status: {status}, Heart Rate: {hr:.1f}, PERCLOS: {perclos:.2f}"
                cv2.putText(frame, message, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

                initial_sent = True
                last_sent_time = time.time()
                f = f[150:]  # 150 프레임만 유지
                closed_frame = 0  # 보내면 감은 프레임 초기화
                total_frame = 0

            # 12초마다 최근 450 프레임 전송 (초기값 보냈으면)
            if time.time() - last_sent_time >= 12 and initial_sent:
                perclos = closed_frame / total_frame
                f = f[150:]
                data = f, perclos
                collector_pipe.send(data)
                print('전송완료', len(f), closed_frame)

                status, hr = collector_pipe.recv()
                message = f"Status: {status}, Heart Rate: {hr:.1f}, PERCLOS: {perclos:.2f}"
                cv2.putText(frame, message, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

                last_sent_time = time.time()
                f = f[150:]  # 150 프레임만 유지
                closed_frame = 0  # 보내면 감은 프레임 초기화
                total_frame = 0

            cv2.imshow('Webcam Stream', frame)  # 웹캠 화면 표시

            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
                break




# 졸음 상태 판단 및 경고음 재생 함수-버전1 (메인에서 실행되는게 버전1)
def determine_sleepiness_and_sound_alarm(determine_pipe):
    while True:
        hr_threshold = 0.04
        perclos_threshold1 = 0.15
        perclos_threshold2 = 0.075
        hr, perclos, normal_hr = determine_pipe.recv()

        hr_drop = (normal_hr - hr) / normal_hr
        sleepy_by_hr = hr_drop > hr_threshold
        sleepy_by_perclos1 = perclos >= perclos_threshold1
        sleepy_by_perclos2 = perclos >= perclos_threshold2

        if sleepy_by_hr and sleepy_by_perclos1:
            status = "severe drowsiness"
            determine_pipe.send(status)
            play_sound('alarm/d5.mp3')  # d5 알람
        elif sleepy_by_hr or sleepy_by_perclos1:
            status = "moderate drowsiness"
            determine_pipe.send(status)
            play_sound('alarm/d4.mp3')  # d4 알람
        elif sleepy_by_hr and sleepy_by_perclos2:
            status = "mild drowsiness"
            determine_pipe.send(status)
            play_sound('alarm/d3.mp3')  # d3 알람
        elif sleepy_by_perclos2:
            status = "very mild drowsiness"
            determine_pipe.send(status)
            play_sound('alarm/d2.mp3')  # d2 알람
        else:
            status = "normal"
            determine_pipe.send(status)



# 졸음 상태 판단 및 경고음 재생 함수-버전2 (이름 바꿔서 실행안되도록 해놓음)
def determine_sleepiness_and_sound_alarm2(determine_pipe):
    while True:
        hr, perclos, normal_hr, normal_perclose = determine_pipe.recv()
        hr_drop = (normal_hr - hr) / normal_hr

        if perclos >= normal_perclose + 0.6 and hr_drop >= 0.05:
            status = "severe drowsiness"
            determine_pipe.send(status)
            play_sound('alarm/d5.mp3')
        elif (perclos >= normal_perclose + 0.6) or (perclos >= normal_perclose + 0.6 and hr_drop >= 0.04):
            status = "moderate drowsiness"
            determine_pipe.send(status)
            play_sound('alarm/d4.mp3')
        elif perclos >= normal_perclose + 0.4:
            status = "mild drowsiness"
            determine_pipe.send(status)
            play_sound('alarm/d3.mp3')
        elif perclos >= normal_perclose + 0.3 and hr_drop >= 0.04:
            status = "very mild drowsiness"
            determine_pipe.send(status)
            play_sound('alarm/d2.mp3')
        elif perclos >= normal_perclose + 0.2:
            status = "minimal drowsiness"
            determine_pipe.send(status)
            play_sound('alarm/d1.mp3')
        else:
            status = "normal"
            determine_pipe.send(status)



def process_video_data(processor_pipe, processor_pipe2, model):  # 심박수 계산
    g_hr = []
    g_per = []
    avg_hr = 0
    two_pass = False
    start = time.time()
    while True:
        _ = []
        f, perclos = processor_pipe.recv()
        f = np.array(f)
        print(f.shape)
        _1 = np.full(450, np.nan)
        _1[0:450] = model([f[0:450] / 255])[0]
        hr = get_hr(_1)
        if hr <= 130 or hr >= 50:
            g_hr.append(hr)
        g_per.append(perclos)
        print("심박수: {}bpm, perclos: {}".format(hr, perclos))
        result = "measuring...", hr

        # 2분이 지나면 평균 심박수 구함
        if two_pass:
            """"""
            # determine_sleepiness_and_sound_alarm2인 경우 (main determine_p의 target과 맞출것.)
            determine_data = hr, perclos, avg_hr, avg_pe
            processor_pipe2.send(determine_data)
            sleepiness_level = processor_pipe2.recv()

            """

            # determine_sleepiness_and_sound_alaram인 경우 (main determine_p의 target과 맞출것.)
            determine_data = hr, perclos, avg_hr
            processor_pipe2.send(determine_data)
            sleepiness_level = processor_pipe2.recv()
            """

            result = sleepiness_level, hr
            # 콘솔에 상태 출력 (디버깅 용)
            print(f"Status: {sleepiness_level}, Heart Rate: {hr:.1f}, PERCLOS: {perclos:.2f}")

        processor_pipe.send(result)

        if time.time() - start > 120 and not two_pass:
            avg_hr = sum(g_hr) / len(g_hr)
            avg_pe = sum(g_per) / len(g_per)
            two_pass = True

if __name__ == "__main__":
    model = model.M_1()
    model.build(input_shape=(None, 450, 8, 8, 3))
    model.load_weights('weights/m1.h5')

    collector_pipe, processor_pipe = multiprocessing.Pipe()
    processor_pipe2, determine_pipe = multiprocessing.Pipe()
    collector_p = multiprocessing.Process(name="collector", target=frame_collector, args=(collector_pipe,))
    processor_p = multiprocessing.Process(name="processor", target=process_video_data,
                                          args=(processor_pipe, processor_pipe2, model))
    determine_p = multiprocessing.Process(name="determine", target=determine_sleepiness_and_sound_alarm2,
                                          args=(determine_pipe,))
    collector_p.start()
    processor_p.start()
    determine_p.start()
    print("ESC를 클릭하시면 종료됩니다.")

    while True:
        if not collector_p.is_alive():
            determine_p.kill()
            processor_p.kill()
            break

    print("측정 종료")

    collector_p.join()
    processor_p.join()
    determine_p.join()


