import imageio
import glob
import os
from tqdm import tqdm

def crop_frame(img, p1, p2): # Crop from x1, y1 to x2, y2
    pass

def crop_center(img):
    pass

seqs_data = [
    ['img/abandon_package', 'v1',  2560, 2860, 10, 'gif/abandon_package.gif', crop_center],
    # ['img/car_interaction', 'v1',  0, 1000, 10, 'gif/car_interaction.gif', crop_center],
    ['img/embrace', 'v1',          120, 360, 10, 'gif/embrace.gif', crop_center],
    ['img/facility_door', 'v1',    400, 900, 10, 'gif/facility_door.gif', crop_center],
    # ['img/hand_interaction', 'v1', 0, 100, 10, 'gif/hand_interaction.gif', crop_center],
    ['img/heavy_carry', 'v1',      1200, 1600, 10, 'gif/heavy_carry.gif', crop_center],
    # ['img/laptop', 'v1',           0, 100, 10, 'gif/laptop.gif', crop_center],
    # ['img/object_transfer', 'v1',  0, 1000, 10, 'gif/object_transfer.gif', crop_center],
    # ['img/purchase', 'v1',         0, 1000, 10, 'gif/purchase.gif', crop_center],
    # ['img/reading', 'v1',          0, 1000, 10, 'gif/reading.gif', crop_center],
    # ['img/riding', 'v1',           0, 1000, 10, 'gif/riding.gif', crop_center],
    ['img/running', 'v1',          150, 300, 10, 'gif/running.gif', crop_center],
    ['img/stand_up', 'v1',         400, 850, 10, 'gif/stand_up.gif', crop_center],
    # ['img/talking', 'v1',          0, 1000, 10, 'gif/talking.gif', crop_center],
    ['img/talking_phone', 'v1',    230, 900, 10, 'gif/talking_phone.gif', crop_center],
    # ['img/texting_phone', 'v1',    0, 1000, 10, 'gif/texting_phone.gif', crop_center],
    # ['img/theft', 'v1',            0, 1000, 10, 'gif/theft.gif', crop_center],
]

class Seq:
    def __init__(self, ul):
        self.folder = ul[0]
        self.camera_view = ul[1]
        self.start = ul[2]
        self.end = ul[3]
        self.step = ul[4]
        self.gif_filename = ul[5]
        self.frame_postprocess = ul[6]

seqs = [Seq(v) for v in seqs_data]

def make_gif(seq):
    # filenames = glob.glob('img/talking_phone/v1/*.png')
    # filename = 'img/talking_phone/v1/{frame_id}.png'
    # print(filenames)
    frames = range(seq.start, seq.end, seq.step)
    filename_tpl = os.path.join(seq.folder, seq.camera_view, '%d.png')
    filenames = [filename_tpl % i for i in frames]

    # https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
    images = []
    with imageio.get_writer(seq.gif_filename, mode='I') as writer:
        for filename in tqdm(filenames):
            image = imageio.imread(filename)
            writer.append_data(image)

for seq in seqs:
    make_gif(seq)