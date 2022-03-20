import os
import re
import time
import json
import math
import random
import decimal
import cv2 as cv
import numpy as np
import mediapipe as mp
'''
Using MediaPipe's Holistic model, this module will 
gather the user's sequence of hand and shoulder coordinates.

The shoulder coordinates will be used as the base reference
when normalizing the data.
'''

#remove word_list

class PreprocessingPipeline:
    def __init__(self):
        self.tracker = None
        self.normalizer = None
        self.augmenter = None

    def get_data_live(self, 
            word, 
            data_path, 
            sample_count = 15, 
            frame_count = 18,
            collect_reverse = False
            ):
        stop = False
        self.capture = cv.VideoCapture(0)
        #Check if data_path exists
        if not os.path.isdir(data_path): 
            print(f'{data_path} does not exists!')
            return 
        self._update_wordlist(data_path, word)
        self._pause_capture(5) #Intro so that the user can prepare

        #Start collecting samples
        for sample in range(sample_count):
            samp_path = os.path.join(data_path, word, str(sample))
            if not os.path.isdir(samp_path):
                os.makedirs(samp_path)
            for frame_num in range(frame_count):
                #Have 2 seconds pause per sample
                if frame_num == 0:
                    self._pause_capture(2, f'({word})')
                _, frame = self.capture.read()
                self.tracker.put_text(frame, f'{word} - Sample: #{sample}')
                
                #Gather key coordinates from the frame
                results = self.tracker.detect(frame) 
                keys = self._get_keypoints(results)
                
                self.tracker.draw_landmarks(frame, results, keys[0])
                cv.imshow('Capture', frame)

                #Normalize data
                keys = self._flatten(keys)
                #keys = self.normalizer.normalize_based_on_shoulders(keys)

                #Save
                npy_path = os.path.join(samp_path, f'{frame_num}.npy')
                np.save(npy_path, keys)
                print(f'Saved->{npy_path}')

                if collect_reverse:
                    rev_keys = self._collect_reverse_data(frame)
                    if not os.path.isdir(samp_path + '-r'):
                        os.makedirs(samp_path + '-r')
                    npy_path = os.path.join(samp_path + '-r', f'{frame_num}.npy')
                    np.save(npy_path, rev_keys)
                    print(f'Saved->{npy_path}')

                if cv.waitKey(10) & 0xFF == ord('q'):
                    stop = True
                    break
            if stop:
                print('Process stopped!')
                break
        self.capture.release()
        cv.destroyAllWindows()
        if sample_count > 0:
            print(f'DATA GATHERED FOR {word}.')
    
    def load_data(self, data_path, save_json = True, augment = 0, thres = 0.02):
        words = os.listdir(data_path)
        words.remove('word_list.txt')
        data = {'label_map': [], 'data': [], 'label': []}
        data['label_map'] = words
        for word in words:
            word_path = os.path.join(data_path, word)
            for sample in os.listdir(word_path):
                sample_data = []
                sample_path = os.path.join(word_path, sample)
                frames = os.listdir(sample_path)
                frames.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
                for frame in frames:
                    frame_path = os.path.join(sample_path, frame)
                    keys = np.load(frame_path)
                    keys = self.normalizer.normalize_based_on_shoulders(keys)
                    sample_data.append(keys)
                sample_data = self.normalizer.reverse_data_shape(np.array(sample_data))
                data['data'].append(sample_data.tolist()) 
                data['label'].append(words.index(word))
                for _ in range(augment):
                    aug_data = []
                    for frame in frames:
                        frame_path = os.path.join(sample_path, frame)
                        keys = np.load(frame_path)
                        keys = self.augmenter.move_point(keys, thres = thres)
                        keys = self.normalizer.normalize_based_on_shoulders(np.array(keys))
                        aug_data.append(keys)
                    aug_data = self.normalizer.reverse_data_shape(np.array(aug_data))
                    data['data'].append(aug_data.tolist())
                    data['label'].append(words.index(word))
        if save_json:
            with open(os.path.join(data_path, 'data.json'), 'w') as f:
                json.dump(data, f, indent=4)
                print('data.json is saved.')
        return data

    def _collect_reverse_data(self, frame):
        rev_frame = cv.flip(frame, 1)
        results = self.tracker.detect(rev_frame) 
        keys = self._get_keypoints(results)
        keys = self._flatten(keys)
        return keys

    def _flatten(self, keys):
        return np.concatenate([np.array(part).flatten() for part in keys]) #[6,42,42]

    def _update_wordlist(self, data_path, word):
        word_list_path = os.path.join(data_path, 'word_list.txt')
        if not os.path.isfile(word_list_path):
            with open(word_list_path, 'w') as _:
                pass
        with open(word_list_path) as wl:
            current_words = [w.rstrip() for w in wl]
        with open(word_list_path, 'a') as wl:
            if word not in current_words:
                wl.write(word + '\n') #Add word to word list
            else:
                print(f'Overriding data for {word}')

    def _get_keypoints(self, results):
        #Get shoulder coordinates
        shoulder_points = [12, 0 , 11] #[right shoulder, head, left shoulder]
        pose = [[res.x, res.y] for res in results.pose_landmarks.landmark] if results.pose_landmarks else np.zeros((33,2))
        shoulders = [pose[p] for p in shoulder_points]
        #Get hand coordinates
        lhand = [[res.x, res.y] for res in results.left_hand_landmarks.landmark] if results.left_hand_landmarks else np.zeros((21,2))
        rhand = [[res.x, res.y] for res in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else np.zeros((21,2))
        return [shoulders, lhand, rhand]

    def _pause_capture(self, seconds, text = ''):
        st = time.time()
        elapsed = 0
        while elapsed < seconds:
            _, frame = self.capture.read()
            et = time.time()
            elapsed = et - st
            self.tracker.put_text(frame, f'Start capture {text} - {int(seconds - elapsed)}s')
            cv.imshow('Capture', frame)
            cv.waitKey(10)

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
        #return [0 if point < 0 or point > 1 else point for point in data]

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

class Augmenter:
    def __init__(self):
        pass

    def move_point(self, data, thres = 0.02):
        return [point + float(random.randrange(-thres*1000, thres*1000))/1000 for point in data]


class Tracker:
    def __init__(self, 
            detect_conf = 0.5, 
            tracking_conf = 0.5
            ):
        self.drawing = mp.solutions.drawing_utils
        self.holistic = mp.solutions.holistic
        self.holistic_model = self.holistic.Holistic(
            min_detection_confidence = detect_conf, 
            min_tracking_confidence = tracking_conf)

    #This method predicts and returns the tracked coordinates in the frame
    def detect(self, frame, color = cv.COLOR_BGR2RGB):
        return self.holistic_model.process(cv.cvtColor(frame, color))

    #Frame modifications methods
    def put_text(self,
            frame, 
            text, 
            position = (12, 25),
            color = (0,0,0)
            ):
        cv.putText(
                frame, text, position, 
                cv.FONT_HERSHEY_SIMPLEX, 
                1, color, 2, 
                cv.LINE_AA)

    def draw_landmarks(self, frame, results, shoulder_points):
        self._draw_shoulder_points(frame, shoulder_points)
        self._draw_hands(frame, results.left_hand_landmarks)
        self._draw_hands(frame, results.right_hand_landmarks)

    def _draw_shoulder_points(self, frame, shoulder_points):
        h, w, c = frame.shape
        for points in shoulder_points:
            x, y = int(points[0]*w), int(points[1]*h)
            cv.circle(frame, (x, y), 5, (0,255,0), cv.FILLED)

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
