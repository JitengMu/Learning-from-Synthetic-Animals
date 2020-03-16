# utility functions for visualization, such as recording an animated figure as a video
import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
 
# from: http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
def fig2data1( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    print(w, h)
    buf = numpy.fromstring ( fig.canvas.tostring_argb(), dtype=numpy.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll ( buf, 3, axis = 2 )
    return buf

def fig2data(fig):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    # fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    canvas.draw()       # draw the canvas, cache the renderer
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)
    image = image.reshape(height, width, 3)
    # print(image.shape)
    return image

def make_video(filename, fps, w, h):
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    vid_output = cv2.VideoWriter()
    vid_output.open(filename, fourcc, fps, (w, h))
    return vid_output

class FigureVideoRecorder:
    ''' Record figure as a video '''
    def __init__(self, filename):
        self.vid = make_video(filename, 30, 640, 480)

    def read_fig(self, fig):
        # read numpy data from a figure
        # http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
        frame = fig2data(fig)
        return frame

    def write_figure(self, fig):
        frame = self.read_fig(fig)
        self.vid.write(frame)

class FigureImageRecorder:
    def __init__(self):
        self.frame_id = 0

    def read_fig(self, fig):
        frame = fig2data(fig)
        return frame

    def write_figure(self, fig):
        frame = self.read_fig(fig)
        filename = '%d.png' % self.frame_id
        plt.imsave(filename, frame)
        self.frame_id += 1
    
class FigureNativeImageRecorder:
    def __init__(self):
        self.frame_id = 0

    def write_figure(self, fig):
        filename = '%d.png' % self.frame_id
        fig.savefig(filename)
        self.frame_id += 1

def test():
    # recorder = FigureNativeImageRecorder()
    # recorder = FigureImageRecorder()
    recorder = FigureVideoRecorder('fig.avi')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for x in range(0, 100):
        y = x
        ax.plot(x, y, '*')
        recorder.write_figure(fig)

if __name__ == '__main__':
    test()

