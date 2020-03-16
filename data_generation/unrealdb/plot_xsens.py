import sys
sys.path.append('.')
import mocap.vis
from vis import FigureVideoRecorder
from tqdm import tqdm

vis = mocap.vis.XsensVisualizer()
recorder = FigureVideoRecorder('figure.avi')
for frame in tqdm(range(0, 2000, 5)):
    vis.draw_frame(frame)
    recorder.write_figure(vis.canvas.fig)