import argparse

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data loader related
        self.parser.add_argument('--dataroot', type=str, default='./datasets/MSRS', help='path of data')
        self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
        self.parser.add_argument('--batch_size', type=int, default=2, help='batch size')
        self.parser.add_argument('--num_workers', type=int, default=4, help='# of threads for data loader')
        self.parser.add_argument('--crop_size', type=int, default=256, help='crop size for training')

        # training related
        self.parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate for training model')
        self.parser.add_argument('--max_epochs', type=int, default=11, help='number of epochs')
        self.parser.add_argument('--val_freq', type=int, default=1, help='validation frequency (epochs)')
        self.parser.add_argument('--print_freq', type=int, default=10, help='print frequency (iterations)')
        self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--gpu', type=int, default=0, help='GPU id')  
        
        # output related
        self.parser.add_argument('--name', type=str, default='PSFusion', help='folder name to save outputs')
        self.parser.add_argument('--class_nb', type=int, default=9, help='class number for segmentation model')
        self.parser.add_argument('--display_dir', type=str, default='./logs', help='path for saving display results')
        self.parser.add_argument('--result_dir', type=str, default='./results', help='path for saving result images and models')
        self.parser.add_argument('--display_freq', type=int, default=10, help='freq (iteration) of display')
        self.parser.add_argument('--img_save_freq', type=int, default=10, help='freq (epoch) of saving images')
        self.parser.add_argument('--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')
        
    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data loader related
        self.parser.add_argument('--dataroot', type=str, default='./datasets/MSRS/test', help='path of data')
        self.parser.add_argument('--dataname', type=str, default='MSRS', help='name of dataset')
        self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--num_workers', type=int, default=4, help='# of threads for data loader')
        self.parser.add_argument('--crop_size', type=int, default=256, help='crop size for testing')
        
        ## mode related
        self.parser.add_argument('--class_nb', type=int, default=9, help='class number for segmentation model')
        self.parser.add_argument('--resume', type=str, default='./results/PSFusion/checkpoints/best_model.pkl', help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--gpu', type=int, default=0, help='GPU id')
        self.parser.add_argument('--save_seg', action='store_true', help='save segmentation results')
        
        # results related
        self.parser.add_argument('--name', type=str, default='PSFusion', help='folder name to save outputs')
        self.parser.add_argument('--result_dir', type=str, default='./Fusion_results', help='path for saving result images and models')
        self.parser.add_argument('--save_path', type=str, default='./test_results', help='path for saving test results')
        
    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt