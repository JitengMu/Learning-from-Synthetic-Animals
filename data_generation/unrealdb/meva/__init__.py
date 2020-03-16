import glob
import yaml
import os
from tqdm import tqdm
from pprint import pprint
import cv2
import pdb
import pickle

class MEVADataset():
    def __init__(self):
        self.actev_path = './actev-data-repo/annotation/DIVA-phase-2/MEVA/KF1-examples/*/*/'
        self.setup()

    def setup(self):
        cache_file = 'cache.pkl'
        if not os.path.isfile(cache_file):
            # Use the raw data structure, use data path for accessing data
            geom_files = glob.glob(os.path.join(self.actev_path, '*.geom.yml'))
            type_files = glob.glob(os.path.join(self.actev_path, '*.types.yml'))
            act_files = glob.glob(os.path.join(self.actev_path, '*.activities.yml'))
            seq_ids = [os.path.basename(v).replace('.geom.yml', '') for v in geom_files]

            video_files = [os.path.join(v[:10], v[20:22], v + '.avi') for v in seq_ids]
            video_files = [os.path.join('./actev-data-repo/corpora/MEVA/video/KF1/', v) for v in video_files]
            video_files = video_files
            videos = {a:b for a,b in zip(seq_ids, video_files)}

            geom = {}
            for seq_id, geom_file in zip(tqdm(seq_ids), geom_files):
                data = yaml.load(open(geom_file))
                seq = {}
                for row in data:
                    if row.get('geom'):
                        id1 = row['geom']['id1'] # object id
                        # id0 = row['geom']['id0'] # track frame id
                        ts0 = row['geom']['ts0'] # video frame id
                        if not seq.get(id1): seq[id1] = {}
                        seq[id1][ts0] = row
                geom[seq_id] = seq
            
            act = []
            for i, f in enumerate(tqdm(act_files)):
                seq_id = os.path.basename(f).replace('.activities.yml', '')
                # assert(seq_id == seq_ids[i]) # No order guarantee.
                data = yaml.load(open(f))
                for row in data:
                    row['seq_id'] = seq_id
                    if row.get('act'):
                        act.append(row)
            with open(cache_file, 'wb') as f:
                pickle.dump([seq_ids, videos, geom, act], f)
        else:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                seq_ids, videos, geom, act = data

        self.seq_ids = seq_ids
        # self.video_files = video_files
        self.videos = videos
        self.geom = geom
        self.act = act
    
    def get_geom(self, seq_id, obj_id, frame_id):
        return self.geom[seq_id][obj_id].get(frame_id)
    
    def get_video(self, seq_id):
        video_file = self.videos[seq_id]
        if not os.path.isfile(video_file):
            print('File %s for %s not exist' % (video_file, seq_id))
        return video_file

    def list_videos(self):
        # print(self.seq_ids)
        # print(self.video_files)
        # vids = []
        # for video_file in self.video_files:
        #     vid = cv2.VideoCapture()
        #     vid.open(video_file)
        #     vids.append(vid)
        #     frame = vid.read()
        return self.video_files

    def make_video(self, filename, fps, w, h):
        # https://www.programcreek.com/python/example/72134/cv2.VideoWriter
        # fourcc = cv2.cv.CV_FOURCC(*'XVID')
        # https://docs.opencv.org/4.0.0/dd/d9e/classcv_1_1VideoWriter.html
        fourcc = cv2.VideoWriter.fourcc(*'XVID')
        # frame_rate = vid_input.get(flag)
        # frame_rate = vid_input.get(cv2.VideoCaptureProperties.CAP_PROP_FPS)
        # https://www.learnopencv.com/how-to-find-frame-rate-or-frames-per-second-fps-in-opencv-python-cpp/
        vid_output = cv2.VideoWriter()
        vid_output.open(filename, fourcc, fps, (w, h))

        return vid_output
    
    # def clip(self, video_file, start, end, output_file):
    #     vid_input = cv2.VideoCapture()
    #     if not os.path.isfile(video_file):
    #         print('Can not find input video file %s' % video_file)
    #         return

    #     vid_input.open(video_file)
    #     # https://docs.opencv.org/4.0.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    #     vid_input.set(cv2.CAP_PROP_POS_FRAMES, start) # CAP_PROP_POS_FRAMES 
    #     frame_rate = vid_input.get(cv2.CAP_PROP_FPS)
    #     w = 640
    #     h = 480
    #     vid_output = self.make_video(output_file, frame_rate, w, h)

    #     for frame_index in range(start, end):
    #         retval, frame = vid_input.read()
    #         frame = cv2.resize(frame, (w, h))
    #         print(frame_index, retval, frame.shape)
    #         vid_output.write(frame)
    #     vid_output.release()

    def clip_act_bb(self, act, output_file):
        seq_id = act['seq_id']
        vid_file = self.get_video(seq_id)
        start, end = act['act']['timespan'][0]['tsr0']

        vid_input = cv2.VideoCapture()
        video_file = self.get_video(act['seq_id'])
        vid_input.open(video_file)
        start, end = act['act']['timespan'][0]['tsr0']
        vid_input.set(cv2.CAP_PROP_POS_FRAMES, start) # CAP_PROP_POS_FRAMES 
        frame_rate = vid_input.get(cv2.CAP_PROP_FPS)

        w = vid_input.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = vid_input.get(cv2.CAP_PROP_FRAME_HEIGHT) 
        w = int(w / 2)
        h = int(h / 2)
        vid_output = self.make_video(output_file, frame_rate, w, h)

        actors = act['act']['actors']
        actors = [v['id1'] for v in actors]

        for frame_id in range(start, end):
            retval, frame = vid_input.read()

            for obj_id in actors:
                geom = self.get_geom(seq_id, obj_id, frame_id)
                if geom: 
                    bbox = [int(v) for v in geom['geom']['g0'].split(' ')]
                    # print(seq_id, obj_id, frame_id)
                    # print(bbox)
                    frame = self.draw_bb(frame, bbox)

            frame = cv2.resize(frame, (w, h))
            if frame is not None:
            # print(frame_id, retval, frame.shape)
                vid_output.write(frame)
            else:
                print('Can not resize frame ', act)
        vid_output.release()
        vid_input.release()


    def unique_act(self):
        act_names = [list(v['act']['act3'].keys())[0] for v in self.act]
        unique_act_names = sorted(list(set(act_names)))
        counts = [(v, act_names.count(v)) for v in unique_act_names]
        counts = sorted(counts, key=lambda v: v[1])
        return unique_act_names

    def query_by_type(self, act_type):
        acts = [v for v in self.act if list(v['act']['act3'].keys())[0] == act_type]

        return acts

    def draw_bb(self, frame, bb):
        frame = cv2.rectangle(frame, 
            (bb[0], bb[1]), (bb[2], bb[3]),
            color = (255, 0, 0),
            thickness=5
        )
        return frame

    def crop_bb(self, frame, bb):
        pass

def clip_all_act():
    dataset = MEVADataset()
    
    act_types = dataset.unique_act()

    acts = []
    for act_type in act_types:
        acts += dataset.query_by_type(act_type)
    
    for act in tqdm(acts):
        act_type = list(act['act']['act3'].keys())[0]
        seq_id = act['seq_id']
        act_id = act['act']['id2']
        subfolder = 'clips/{act_type}'.format(**locals())
        if not os.path.isdir(subfolder): os.makedirs(subfolder)
        filename = os.path.join(subfolder, '{seq_id}_{act_id}.avi'.format(**locals()))

        if not os.path.isfile(filename):
            dataset.clip_act_bb(act, filename)

def list_all_act():
    dataset = MEVADataset()
    # pprint(dataset.act)
    # pprint(dataset.unique_act())
    act_types = dataset.unique_act()
    act_type = act_types[0]
    print(act_type)
    acts = dataset.query_by_type(act_type)
    act = acts[0]
    print(act)
    # vid_file = dataset.get_video(act['seq_id'])
    # start, end = act['act']['timespan'][0]['tsr0']
    # dataset.clip(vid_file, start, end, 'clip.avi')
    dataset.clip_act_bb(act, 'clip.avi')


    
    # pprint(dataset.list_videos())
    # video_files = dataset.list_videos()

    # dataset.clip(video_files[0], 0, 100, 'test.avi')



def test():
    # list_all_act()
    clip_all_act()
    # dataset = MEVADataset()
    # dataset.list_annotation()

