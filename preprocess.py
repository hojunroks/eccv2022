import os
from PIL import Image
from argparse import ArgumentParser


class Preprocess:
    def __init__(self, orig_data_dir, preprocessed_data_dir):
        self.o = orig_data_dir
        self.p = preprocessed_data_dir


    def apply(self, method):
        do = lambda x: x
        if method == 'resize64':
            do = lambda img: self.resize(img, 64)
        elif method == 'resize128':
            do = lambda img: self.resize(img, 128)

        for f in os.listdir(self.o):
            i = Image.open(os.path.join(self.o, f))
            processed = do(i)
            processed.save(os.path.join(self.p, f))


    def resize(self, img, size):
        return img.resize((size, size))

                
                
                
                
                
def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data_preprocessed/celeba/img_align_celeba')
    parser.add_argument('--new_dir', type=str, default='data_preprocessed/celeba/img_align_celeba_64')
    parser.add_argument('--method', type=str, default='resize64')

    args = parser.parse_args()
    p = Preprocess(args.data_dir, args.new_dir)
    p.apply(args.method)





if __name__ =='__main__':
    main()
