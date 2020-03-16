import argparse
import os.path as osp
class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="adaptive segmentation netowork")
        parser.add_argument("--animal", type=str, default='horse',help="horse or tiger")
        parser.add_argument("--model", type=str, default='hg',help="available options : hg")
        parser.add_argument("--source", type=str, default='synthetic_animal_sp',help="source dataset : loading synthetic data gt labels")
        parser.add_argument("--target-ssl", type=str, default='real_animal_crop',help="target semi-supervised learning dataset : loading generated pseudo-labels")
        parser.add_argument("--target", type=str, default='real_animal_sp',help="target dataset : loading target data ground-truth labels")
        parser.add_argument("--batch-size", type=int, default=6, help="input batch size.")
        parser.add_argument("--workers", type=int, default=4, help="number of threads.")
        parser.add_argument("--data-label-folder-target", type=str, default=None, help="Path to the soft assignments in the target dataset.")    
        parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="initial learning rate for the segmentation network.")
        parser.add_argument("--momentum", type=float, default=0.0, help="Momentum component of the optimiser.")
        parser.add_argument("--weight-decay", type=float, default=0.000, help="Regularisation parameter for L2-loss.")
        parser.add_argument("--num-classes", type=int, default=18, help="Number of classes of keypoints.")
        parser.add_argument("--num-epochs", type=int, default=60, help="Number of training epochs.")
        parser.add_argument("--init-weights", type=str, default=None, help="initial model.")
        parser.add_argument("--print-freq", type=int, default=100, help="print loss and time fequency.")
        parser.add_argument("--checkpoint", type=str, default='/path/to/snapshots/', help="Where to save snapshots of the model.")
        parser.add_argument("--gamma_", type=float, default=10.0, help="target dataset loss coefficient")

        # stacked hourglass
        parser.add_argument('-f', '--flip', dest='flip', action='store_true', help='flip the input during validation')
        parser.add_argument('--inp-res', default=256, type=int, help='input resolution (default: 256)')
        parser.add_argument('--out-res', default=64, type=int, help='output resolution (default: 64, to gen GT)')
        parser.add_argument('--scale-factor', type=float, default=0.25, help='Scale factor (data aug).')
        parser.add_argument('--rot-factor', type=float, default=30, help='Rotation factor (data aug).')
        parser.add_argument('--sigma', type=float, default=1, help='Groundtruth Gaussian sigma.')
        parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian', choices=['Gaussian', 'Cauchy'], help='Labelmap dist type: (default=Gaussian)')
        parser.add_argument('--image-path', default='/home/jiteng/data/animal_data/', type=str, help='path to images')
        parser.add_argument('--arch', '-a', metavar='ARCH', default='hg', help='model architecture: (default: hg)')
        parser.add_argument('-s', '--stacks', default=4, type=int, metavar='N', help='Number of hourglasses to stack')
        parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N', help='Number of residual modules at each location in the hourglass')
        parser.add_argument('--resnet-layers', default=50, type=int, metavar='N', help='Number of resnet layers', choices=[18, 34, 50, 101, 152])
        parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
        return parser.parse_args()
    
    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
    
        # save to the disk
        file_name = osp.join(args.checkpoint, 'opt.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')    
        
