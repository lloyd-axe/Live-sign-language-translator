import os
import re
import time
import json
import math
import random
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
'''
Using MediaPipe's Holistic model, this module will 
gather the user's sequence of hand and shoulder coordinates.

The head and shoulder points will be used to normalize all collected data
The main idea is to form a triangle using the head and shoulder coordinates.
Get the centroid of the triangle and align this point to the center of the screen.
After that, adjust all other coordinates accordingly.
This technique will normalize the position of the data.

Another technique is to set a base length for the user's shoulder.
If the user is far or near, all the collected points will be scaled according
to the base legth of the shoulder. This technique will normalize the scale of the data.
'''

class PreprocessingPipeline:
    def __init__(self):
        self.cv = None
        self.normalizer = None
        self.augmenter = None

    def collect_live_data(self, 
            word, 
            data_path, 
            sample_count = 15, 
            frame_count = 18,
            collect_reverse = False
            ):
        stop = False
        capture = cv2.VideoCapture(0)
        #Check if data_path exists
        if not os.path.isdir(data_path): 
            print(f'{data_path} does not exists!')
            return 
        self.cv.pause_capture(capture,5) #Intro so that the user can prepare

        #Start collecting samples
        for sample in range(sample_count):
            samp_path = os.path.join(data_path, word, str(sample))
            self._create_save_dir(samp_path)
            for frame_num in range(frame_count):
                #Have 2 seconds pause per sample
                if frame_num == 0:
                    self.cv.pause_capture(capture, 2, f'({word})')
                frame, keys = self.cv.capture_collect(capture, draw = True, text = f'{word} - Sample: #{sample}')
                keys = self.normalizer.flatten(keys)

                self._save_npy(keys,samp_path, frame_num)

                if collect_reverse:
                    #This is to omit collecting data using both hands
                    rev_keys = self.cv.collect_reverse_data(frame)
                    rev_keys = self.normalizer.flatten(rev_keys)
                    self._create_save_dir(samp_path + '-r')
                    self._save_npy(rev_keys, samp_path + '-r', frame_num)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    stop = True
                    break
            if stop:
                print('Process stopped!')
                break
        capture.release()
        cv2.destroyAllWindows()
        if sample_count > 0:
            print(f'DATA GATHERED FOR {word}.')
    
    def load_data(self, data_path, save_json = True, augment = 0, thres = 0.02):
        #This load method will include normalization and augmentation of the data
        words = os.listdir(data_path)
        data = {'label_map': [], 'data': [], 'label': []}
        data['label_map'] = words
        for word in words:
            word_path = os.path.join(data_path, word)
            for sample in os.listdir(word_path):
                sample_path = os.path.join(word_path, sample)
                frames = os.listdir(sample_path)
                frames.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

                sample_data = self._get_data_from_frames(frames, sample_path)
                data['data'].append(sample_data.tolist()) 
                data['label'].append(words.index(word))
                
                for _ in range(augment):
                    #This is to produce augmented data
                    aug_data = self._get_data_from_frames(frames, sample_path, thres = thres)
                    data['data'].append(aug_data.tolist())
                    data['label'].append(words.index(word))
        if save_json:
            with open(os.path.join(data_path, 'data.json'), 'w') as f:
                json.dump(data, f, indent=4)
                print('data.json is saved.')
        return data

    def _get_data_from_frames(self, frames, sample_path, thres = 0):
        sample_data = []
        for frame in frames:
            frame_path = os.path.join(sample_path, frame)
            keys = np.load(frame_path)
            if thres > 0:
                keys = self.augmenter.move_points(keys, thres = thres)
            keys = self.normalizer.normalize_based_on_shoulders(keys)
            sample_data.append(keys)
        return self.normalizer.reverse_data_shape(np.array(sample_data))

    def _create_save_dir(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def _save_npy(self, data, samp_path, frame_num):
        npy_path = os.path.join(samp_path, f'{frame_num}.npy')
        np.save(npy_path, data)
        print(f'Saved->{npy_path}')

    @staticmethod
    def display_data(data, data_num):
        plt.figure(figsize=(20,8))
        count = 0
        row = 2 if (data_num > 10) else 1
        col = data_num - (data_num//row) if row == 2 else data_num
        for i in range(data_num):
            count += 1
            plt.subplot(1, col, count)
            plt.imshow(data['data'][i], cmap='Greys')
        plt.show()

class Normalizer:
    def __init__(self, base_scale = 0.25):
        self.base_shoulder_scale = base_scale

    def normalize_based_on_shoulders(self, data):
        data = data.reshape(45,2)
        shoulder_points = data[:3]
        adj_dist = self._get_adjustment_distances(shoulder_points)
        adj_data = self._adjust_positions(data, adj_dist)
        scaled_data = self._normalize_scale(adj_data, shoulder_points)
        clean_data = self._clean_data(scaled_data)
        return clean_data

    def _adjust_positions(self, data, adj_dist):
        adj_data = []
        for points in data:
            x_adj = points[0] + adj_dist[0]
            y_adj = points[1] + adj_dist[1]
            adj_data.append([x_adj, y_adj])
        return adj_data

    def _normalize_scale(self, data, shoulder_points):
        scale = self._get_scale(shoulder_points)
        flat_data = np.concatenate([np.array(part).flatten() for part in data])
        scaled_data = [point*scale for point in flat_data]
        return scaled_data

    def _clean_data(self, data):
        separated_data = self._separate_xy(data)
        return (separated_data - np.min(separated_data))/np.ptp(separated_data)

    def _separate_xy(self, data):
        data = np.array(data).reshape(45,2)
        x, y = [], []
        for points in data:
            x.append(points[0])
            y.append(points[1])
        separated_data = np.concatenate([np.array(part).flatten() for part in [x,y]])
        return separated_data

    def _get_scale(self, shoulder_points):
        x0, y0 = shoulder_points[0][0], shoulder_points[0][1]
        x1, y1 = shoulder_points[1][0], shoulder_points[1][1]
        shoulder_dist = math.hypot(x1-x0, y1-y0)
        scale = self.base_shoulder_scale / shoulder_dist
        return scale

    def _get_adjustment_distances(self, shoulder_points):
        centroid = self._get_centroid(shoulder_points)
        x_adj = 0.5 - centroid[0]
        y_adj = 0.5 - centroid[1]
        return x_adj, y_adj

    def _get_centroid(self, shoulder_points):
        x = (shoulder_points[0][0] + shoulder_points[1][0] + shoulder_points[2][0]) / 3
        y = (shoulder_points[0][1] + shoulder_points[1][1] + shoulder_points[2][1]) / 3
        return (x, y)

    def reverse_data_shape(self, data):
        idata = []
        for i in range(data.shape[1]):
            x = []
            for j in range(data.shape[0]):
                x.append(data[j][i])
            idata.append(x)
        idata = np.array(idata)
        return idata
    
    def flatten(self, keys):
        return np.concatenate([np.array(part).flatten() for part in keys]) #[6,42,42]

class Augmenter:
    def __init__(self):
        pass

    def move_points(self, data, thres):
        return np.array([point + float(random.randrange(-thres*1000, thres*1000))/1000 for point in data])

class CVModule:
    def __init__(self, 
            detect_conf = 0.5, 
            tracking_conf = 0.5
            ):
        self.drawing = mp.solutions.drawing_utils
        self.holistic = mp.solutions.holistic
        self.holistic_model = self.holistic.Holistic(
            min_detection_confidence = detect_conf, 
            min_tracking_confidence = tracking_conf)

    def capture_collect(self, cap, draw = False, text = ''):
        _, frame = cap.read()
        #Gather key coordinates from the frame
        results = self.detect(frame) 
        keys = self._get_keypoints(results)
        if text != '':
            self.put_text(frame, text)
        if draw:
            self.draw_landmarks(frame, results, keys[0])
        cv2.imshow('Capture', frame)
        return frame, keys
    
    def collect_reverse_data(self, frame):
        rev_frame = cv2.flip(frame, 1)
        results = self.detect(rev_frame) 
        keys = self._get_keypoints(results)
        return keys

    def pause_capture(self, cap, seconds, text = ''):
        st = time.time()
        elapsed = 0
        while elapsed < seconds:
            _, frame = cap.read()
            et = time.time()
            elapsed = et - st
            self.put_text(frame, f'Start capture {text} - {int(seconds - elapsed)}s')
            cv2.imshow('Capture', frame)
            cv2.waitKey(10)

    def _get_keypoints(self, results):
        #Get shoulder coordinates
        shoulder_points = [12, 0 , 11] #[right shoulder, head, left shoulder]
        pose = [[res.x, res.y] for res in results.pose_landmarks.landmark] if results.pose_landmarks else np.zeros((33,2))
        shoulders = [pose[p] for p in shoulder_points]
        #Get hand coordinates
        lhand = [[res.x, res.y] for res in results.left_hand_landmarks.landmark] if results.left_hand_landmarks else np.zeros((21,2))
        rhand = [[res.x, res.y] for res in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else np.zeros((21,2))
        return [shoulders, lhand, rhand]

    def detect(self, frame, color = cv2.COLOR_BGR2RGB):
        return self.holistic_model.process(cv2.cvtColor(frame, color))

    def put_text(self,
            frame, 
            text, 
            position = (12, 25),
            color = (0,0,0)
            ):
        cv2.putText(
                frame, text, position, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, color, 2, 
                cv2.LINE_AA)

    def draw_landmarks(self, frame, results, shoulder_points):
        self._draw_shoulder_points(frame, shoulder_points)
        self._draw_hands(frame, results.left_hand_landmarks)
        self._draw_hands(frame, results.right_hand_landmarks)

    def _draw_shoulder_points(self, frame, shoulder_points):
        h, w, c = frame.shape
        for points in shoulder_points:
            x, y = int(points[0]*w), int(points[1]*h)
            cv2.circle(frame, (x, y), 5, (0,255,0), cv2.FILLED)

    def _draw_hands(self, frame, hand_landmarks):
        self.drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.holistic.HAND_CONNECTIONS,
                self.drawing.DrawingSpec(
                color = (0,255,0), 
                thickness = 1, 
                circle_radius = 2),
                self.drawing.DrawingSpec(
                    color = (255,255,255), 
                    thickness = 1, 
                    circle_radius = 1))
