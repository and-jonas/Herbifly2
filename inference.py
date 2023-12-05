
import glob
from ImageSegmentor import Segmentor
import os

os.getcwd()

base_dir = "/home/anjonas/kp-public/Evaluation/Projects/KP0011/7/handheld"


def run():
    dirs_to_process = [base_dir]  # must be a list  # ALL DATA
    dir_output = f'{base_dir}/Output_vegann'
    dir_vegetation_model = "vegAnn_herbifly.pt"
    dir_col_model = "segcol_rf.pkl"
    dir_patch_coordinates = None
    image_pre_segmentor = Segmentor(dirs_to_process=dirs_to_process,
                                    dir_vegetation_model=dir_vegetation_model,
                                    dir_col_model=dir_col_model,
                                    dir_patch_coordinates=dir_patch_coordinates,
                                    dir_output=dir_output,
                                    overwrite=False,
                                    img_type="JPG")
    image_pre_segmentor.process_images()


if __name__ == "__main__":
    run()
