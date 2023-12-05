import os
import torch
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import copy
import pickle
import imageio
import cv2
import flash
from flash.image import SemanticSegmentation, SemanticSegmentationData
from transforms2 import set_input_transform_options
from multiprocessing import Manager, Process
import SegmentationFunctions
import utils


transform = set_input_transform_options(
    train_size=512,
    crop_factor=0.688,
    p_color_jitter=0,
    blur_kernel_size=1,
    predict_size=(4000, 6000),
    predict_scale=0.5
)


class Segmentor:

    def __init__(self, dirs_to_process, dir_patch_coordinates, dir_output, dir_vegetation_model,
                 dir_col_model,
                 overwrite,
                 img_type):
        self.dirs_to_process = dirs_to_process
        self.dir_patch_coordinates = Path(dir_patch_coordinates) if dir_patch_coordinates is not None else None
        self.dir_vegetation_model = dir_vegetation_model
        self.dir_col_model = dir_col_model
        self.overwrite = overwrite
        # output paths
        self.path_output = Path(dir_output)
        self.path_mask = self.path_output / "SegVeg" / "Mask"
        self.path_overlay = self.path_output / "SegVeg" / "Overlay"
        self.path_col_mask = self.path_output / "SegVeg" / "ColMask"
        self.image_type = img_type
        # load the segmentation models
        self.vegetation_model = SemanticSegmentation.load_from_checkpoint(self.dir_vegetation_model)
        with open(self.dir_col_model, 'rb') as model:
            self.col_model = pickle.load(model)
        # instantiate trainer
        # self.trainer = flash.Trainer(max_epochs=1, accelerator='gpu', devices=[0])
        self.trainer = flash.Trainer(max_epochs=1, accelerator='cpu')

    def prepare_workspace(self):
        """
        Creates all required output directories
        """
        for path in [self.path_output, self.path_mask, self.path_overlay, self.path_col_mask]:
            path.mkdir(parents=True, exist_ok=True)

    def file_feed(self):
        """
        Creates a list of paths to images that are to be processed
        :param img_type: a character string, the file extension, e.g. "JPG"
        :return: paths
        """
        # get all files and their paths
        files = []
        for d in self.dirs_to_process:
            files.extend(glob.glob(f'{d}/*.{self.image_type}'))
        # removes all Reference images
        files = [f for f in files if "Ref" not in f]
        # removes all processed images
        if not self.overwrite:
            processed = glob.glob(f'{self.path_col_mask}/*.png')
            existing = [os.path.basename(x).replace(".png", "") for x in processed]
            files_to_proc = [os.path.basename(f).replace(".JPG", "") for f in files]
            idx = [idx for idx, img in enumerate(files_to_proc) if img not in existing]
            files = [files[i] for i in idx]

        return files

    @staticmethod
    def make_overlay(patch, mask, colors=[(1, 0, 0, 0.25)]):
        img_ = Image.fromarray(patch, mode="RGB")
        img_ = img_.convert("RGBA")
        class_labels = np.unique(mask)
        for i, v in enumerate(class_labels[1:]):
            r, g, b, a = colors[i]
            M = np.where(mask == v, 255, 0)
            M = M.ravel()
            M = np.expand_dims(M, -1)
            out_mask = np.dot(M, np.array([[r, g, b, a]]))
            out_mask = np.reshape(out_mask, newshape=(patch.shape[0], patch.shape[1], 4))
            out_mask = out_mask.astype("uint8")
            M = Image.fromarray(out_mask, mode="RGBA")
            img_.paste(M, (0, 0), M)
        img_ = img_.convert('RGB')
        overlay = np.asarray(img_)

        return overlay

    def segment_image(self, patch, model, transform, colors):
        """
        Segments an image using a pre-trained semantic segmentation model.
        Creates probability maps, binary segmentation masks, and overlay
        :param image: The image to be processed as an numpy array.
        :param coordinates: A tuple of coordinates defining the ROI.
        :return: The resulting binary segmentation mask.
        """

        # image axes must be re-arranged
        patch_ = np.moveaxis(patch, 2, 0) / 255.0

        # create a datamodule from numpy array
        datamodule = SemanticSegmentationData.from_numpy(
            predict_data=[patch_],
            num_classes=2,
            train_transform=transform,
            val_transform=transform,
            test_transform=transform,
            predict_transform=transform,
            batch_size=1,  # required
        )

        # make predictions
        print("starting prediction")
        predictions = self.trainer.predict(
            model=model,
            datamodule=datamodule,
        )

        # extract predictions
        predictions = predictions[0][0]['preds']

        # transform predictions to probabilities and labels
        probabilities = torch.softmax(predictions, dim=0)
        probabilities_ear = probabilities[0]
        mask = torch.argmax(probabilities, dim=0)
        mask_8bit = np.uint8((mask*255) / (len(np.unique(mask))-1))

        overlay = self.make_overlay(patch, mask_8bit, colors=colors)

        return probabilities_ear, np.asarray(mask_8bit), overlay

    def process_images(self):
        """
        Wrapper, processing all images
        """
        self.prepare_workspace()
        files = self.file_feed()

        for file in files:

            # read image
            base_name = os.path.basename(file)
            stem_name = Path(file).stem
            png_name = base_name.replace("." + self.image_type, ".png")
            img = Image.open(file)
            pix = np.array(img)

            # sample patch from image using coordinate file
            if self.dir_patch_coordinates is not None:
                c = pd.read_table(f'{self.dir_patch_coordinates}/{stem_name}.txt', sep=",").iloc[0, :].tolist()
                patch = pix[c[2]:c[3], c[0]:c[1]]
            else:
                patch = pix

            # imageio.imwrite(self.path_patch / png_name, patch)

            # (2) segment vegetation ===================================================================================
            proba, pred_mask, overlay = self.segment_image(
                patch,
                model=self.vegetation_model,
                transform=transform,
                colors=[(0, 0, 1, 0.25)]
            )

            # output paths
            mask_name = self.path_mask / png_name
            overlay_name = self.path_overlay / base_name

            # save output
            imageio.imwrite(mask_name, pred_mask)
            imageio.imwrite(overlay_name, overlay)

            # (3) color-based segmentation =============================================================================

            # downscale
            x_new = int(patch.shape[0] * (1 / 2))
            y_new = int(patch.shape[1] * (1 / 2))
            patch_seg = cv2.resize(patch, (y_new, x_new), interpolation=cv2.INTER_LINEAR)

            # extract pixel features
            color_spaces, descriptors, descriptor_names = SegmentationFunctions.get_color_spaces(patch_seg)
            descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])

            # get pixel label probabilities
            segmented_flatten_probs = self.col_model.predict(descriptors_flatten)

            # restore image
            preds = segmented_flatten_probs.reshape((descriptors.shape[0], descriptors.shape[1]))

            # convert to mask
            mask = np.zeros_like(patch_seg)
            mask[np.where(preds == "brown")] = (102, 61, 20)
            mask[np.where(preds == "yellow")] = (255, 204, 0)
            mask[np.where(preds == "green")] = (0, 100, 0)

            # upscale
            x_new = int(patch_seg.shape[0] * (2))
            y_new = int(patch_seg.shape[1] * (2))
            mask = cv2.resize(mask, (y_new, x_new), interpolation=cv2.INTER_NEAREST)

            # remove background
            col_mask_name = self.path_col_mask / png_name
            col_mask = copy.copy(mask)
            col_mask[np.where(pred_mask == 0)] = (0, 0, 0)
            imageio.imwrite(col_mask_name, col_mask)


class ImagePostSegmentor:

    def __init__(self,
                 base_dir,
                 dirs_to_process,
                 dir_veg_masks,
                 dir_patch_coordinates,
                 dir_stem_ear_masks, dir_output, dir_model, img_type, mask_type, overwrite,
                 save_masked_images, save_color_masks,
                 n_cpus):
        self.path_base_dir = Path(base_dir)
        self.dirs_to_process = dirs_to_process
        self.dir_patch_coordinates = dir_patch_coordinates
        self.dir_veg_masks = dir_veg_masks
        self.dir_stem_ear_masks = dir_stem_ear_masks
        self.dir_model = Path(dir_model)
        # output paths
        self.path_output = Path(dir_output)
        self.path_mask = self.path_output / "Mask"
        # - color masks
        self.patch_mask_ear = self.path_output / "EarMask"
        self.patch_mask_stem = self.path_output / "StemMask"
        self.patch_mask_veg = self.path_output / "VegMask"
        self.patch_mask_veg_no_ear_no_stem = self.path_output / "VegNoEarNoStemMask"
        # - image masks
        self.patch_img_veg = self.path_base_dir / "SegImg" / "VegImg"
        self.patch_img_stem = self.path_base_dir/ "SegImg" / "StemImg"
        self.patch_img_ear = self.path_base_dir / "SegImg" / "EarImg"
        self.patch_img_veg_no_ear_no_stem = self.path_base_dir / "SegImg" / "VegNoEarNoStemImg"
        # - csv
        self.path_stats = self.path_output / "Stats"
        # helpers
        self.image_type = img_type
        self.mask_type = mask_type
        self.overwrite = overwrite
        self.save_masked_images = save_masked_images
        self.save_color_masks = save_color_masks
        # settings
        self.n_cpus = n_cpus
        # load model
        with open(self.dir_model, 'rb') as model:
            self.model = pickle.load(model)

        model.n_jobs = 1  # disable parallel; parallelize over images instead

    def prepare_workspace(self):
        """
        Creates all required output directories
        """
        for path in [
            self.path_output, self.path_mask, self.patch_mask_ear, self.patch_mask_veg, self.patch_mask_stem,
            self.patch_mask_veg_no_ear_no_stem, self.patch_img_veg, self.patch_img_ear, self.patch_img_stem,
            self.patch_img_veg_no_ear_no_stem, self.path_stats
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def file_feed(self):
        """
        Creates a list of paths to images that are to be processed
        :param img_type: a character string, the file extension, e.g. "JPG"
        :return: paths
        """

        # get all files and their paths
        files = []
        for d in self.dirs_to_process:
            files.extend(glob.glob(f'{d}/*.{self.image_type}'))
        # removes all Reference images
        files = [f for f in files if "Ref" not in f]
        return files

    def segment_image(self, img, model):
        """
        Segments an image using a pre-trained pixel classification model.
        Creates probability maps, binary segmentation masks, and overlay
        :param veg_mask: vegetation mask (ground truth or flash predictions)
        :param img: The image to be processed.
        :return: The resulting binary segmentation mask.
        """

        model.n_jobs = 1  # disable parallel; parallelize over images instead

        # extract pixel features
        color_spaces, descriptors, descriptor_names = SegmentationFunctions.get_color_spaces(img)
        descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])

        # extract pixel label probabilities
        segmented_flatten_probs = model.predict(descriptors_flatten)

        # restore image
        preds = segmented_flatten_probs.reshape((descriptors.shape[0], descriptors.shape[1]))

        # convert to mask
        mask = np.zeros_like(img)
        mask[np.where(preds == "brown")] = (102, 61, 20)
        mask[np.where(preds == "yellow")] = (255, 204, 0)
        mask[np.where(preds == "green")] = (0, 100, 0)

        # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        # axs[0].imshow(mask)
        # axs[0].set_title('img')
        # axs[1].imshow(img)
        # axs[1].set_title('orig_mask')
        # plt.show(block=True)

        return mask

    def process_image(self, work_queue, result):

        for job in iter(work_queue.get, 'STOP'):

            file = job['file']

            # read image
            base_name = os.path.basename(file)
            stem_name = Path(file).stem
            png_name = base_name.replace("." + self.image_type, ".png")
            csv_name = base_name.replace("." + self.image_type, ".csv")

            img = Image.open(file)
            pix = np.array(img)

            # sample patch from image
            c = pd.read_table(f'{self.dir_patch_coordinates}/{stem_name}.txt', sep=",").iloc[0, :].tolist()
            patch = pix[c[2]:c[3], c[0]:c[1]]

            # downscale
            x_new = int(patch.shape[0] * (1 / 2))
            y_new = int(patch.shape[1] * (1 / 2))
            patch_seg = cv2.resize(patch, (y_new, x_new), interpolation=cv2.INTER_LINEAR)

            # output paths
            mask_name = self.path_mask / png_name

            if not self.overwrite and os.path.exists(mask_name):
                continue

            # segment the entire patch
            mask = self.segment_image(patch_seg, self.model)

            # upscale
            x_new = int(patch_seg.shape[0] * (2))
            y_new = int(patch_seg.shape[1] * (2))
            mask = cv2.resize(mask, (y_new, x_new), interpolation=cv2.INTER_NEAREST)

            # ==========================================================================================================

            # load segmentation masks
            veg_mask = imageio.imread(f'{self.dir_veg_masks}/{png_name}')
            stem_ear_mask = imageio.imread(f'{self.dir_stem_ear_masks}/{png_name}')

            # ear mask
            ear_mask = copy.copy(stem_ear_mask)
            ear_mask[np.where(stem_ear_mask == 255)] = 0
            ear_mask[np.where(stem_ear_mask == 127)] = 255

            # stem mask
            stem_mask = copy.copy(stem_ear_mask)
            stem_mask[np.where(stem_ear_mask == 127)] = 0

            # veg mask
            veg_no_stem_ear_mask = copy.copy(veg_mask)
            veg_no_stem_ear_mask[np.where(stem_mask == 255)] = 0
            veg_no_stem_ear_mask[np.where(ear_mask == 255)] = 0

            # ==========================================================================================================

            # remove background and/or objects - color masks
            veg_col_mask = copy.copy(mask)
            veg_col_mask[np.where(veg_mask == 0)] = (0, 0, 0)

            ear_col_mask = copy.copy(mask)
            ear_col_mask[np.where(ear_mask != 255)] = (0, 0, 0)  # removes background (veg and and soil)

            stem_col_mask = copy.copy(mask)
            stem_col_mask[np.where(stem_mask != 255)] = (0, 0, 0)  # removes background (veg and and soil)

            veg_col_mask_no_ear_no_stem = copy.copy(veg_col_mask)
            veg_col_mask_no_ear_no_stem[np.where(veg_no_stem_ear_mask != 255)] = (0, 0, 0)

            # remove background and/or objects - original patches
            veg_image = copy.copy(patch)
            veg_image[np.where(veg_mask == 0)] = (0, 0, 0)

            ear_image = copy.copy(patch)
            ear_image[np.where(ear_mask == 0)] = (0, 0, 0)

            stem_image = copy.copy(patch)
            stem_image[np.where(stem_mask == 0)] = (0, 0, 0)

            veg_no_ear_no_stem_image = copy.copy(veg_image)
            veg_no_ear_no_stem_image[np.where(ear_mask == 255)] = (0, 0, 0)
            veg_no_ear_no_stem_image[np.where(stem_mask == 255)] = (0, 0, 0)

            if self.save_color_masks:
                imageio.imwrite(mask_name, mask)
                imageio.imwrite(self.patch_mask_veg / png_name, veg_col_mask)
                imageio.imwrite(self.patch_mask_ear / png_name, ear_col_mask)
                imageio.imwrite(self.patch_mask_stem / png_name, stem_col_mask)
                imageio.imwrite(self.patch_mask_veg_no_ear_no_stem / png_name, veg_col_mask_no_ear_no_stem)

            if self.save_masked_images:
                imageio.imwrite(self.patch_img_veg / png_name, veg_image)
                imageio.imwrite(self.patch_img_ear / png_name, ear_image)
                imageio.imwrite(self.patch_img_stem / png_name, stem_image)
                imageio.imwrite(self.patch_img_veg_no_ear_no_stem / png_name, veg_no_ear_no_stem_image)

            # ==========================================================================================================

            # get color properties at levels
            levels = ["veg", "ear", "stem", "veg2"]
            level_masks = [veg_mask, ear_mask, stem_mask, veg_no_stem_ear_mask]
            level_images = [veg_image, ear_image, stem_image, veg_no_ear_no_stem_image]

            # iterate over levels (organs)
            stats = []
            stat_names = []
            for l, l_m, l_i in zip(levels, level_masks, level_images):
                desc, desc_names = utils.color_index_transformation(l_i)
                # iterate over color features
                for d, d_n in zip(desc, desc_names):
                    s, n = utils.index_distribution(image=d,
                                                    feature_name=d_n,
                                                    level_id=l,
                                                    level_mask=l_m)
                    stats.extend(s)
                    stat_names.extend(n)
            df = pd.DataFrame([stats], columns=stat_names)
            df.insert(loc=0, column='image_id', value=stem_name)

            # ==========================================================================================================

            # get pixel fractions at levels

            # total cover per fraction
            veg_cover = len(np.where(veg_mask == 255)[0])/(patch.shape[0]*patch.shape[1])
            ear_cover = len(np.where(ear_mask == 255)[0])/(patch.shape[0]*patch.shape[1])
            stem_cover = len(np.where(stem_mask == 255)[0])/(patch.shape[0]*patch.shape[1])
            veg_cover_no_ear_no_stem = len(np.where(veg_no_stem_ear_mask == 255)[0])/(patch.shape[0]*patch.shape[1])
            cover_stat_names = ["veg_cover", "ear_cover", "stem_cover", "veg2_cover"]
            df[cover_stat_names] = [[veg_cover, ear_cover, stem_cover, veg_cover_no_ear_no_stem]]

            # cover within fraction per color
            ear_green = len(np.where(ear_col_mask[:, :, 1] == 100)[0])/len(np.where(ear_mask == 255)[0])
            ear_chlr = len(np.where(ear_col_mask[:, :, 1] == 204)[0])/len(np.where(ear_mask == 255)[0])
            ear_necr = len(np.where(ear_col_mask[:, :, 1] == 61)[0])/len(np.where(ear_mask == 255)[0])
            stem_green = len(np.where(stem_col_mask[:, :, 1] == 100)[0])/len(np.where(stem_mask == 255)[0])
            stem_chlr = len(np.where(stem_col_mask[:, :, 1] == 204)[0])/len(np.where(stem_mask == 255)[0])
            stem_necr = len(np.where(stem_col_mask[:, :, 1] == 61)[0])/len(np.where(stem_mask == 255)[0])
            veg_green = len(np.where(veg_col_mask[:, :, 1] == 100)[0])/len(np.where(veg_mask == 255)[0])
            veg_chlr = len(np.where(veg_col_mask[:, :, 1] == 204)[0])/len(np.where(veg_mask == 255)[0])
            veg_necr = len(np.where(veg_col_mask[:, :, 1] == 61)[0])/len(np.where(veg_mask == 255)[0])
            veg2_green = len(np.where(veg_col_mask_no_ear_no_stem[:, :, 1] == 100)[0])/len(np.where(veg_no_stem_ear_mask == 255)[0])
            veg2_chlr = len(np.where(veg_col_mask_no_ear_no_stem[:, :, 1] == 204)[0])/len(np.where(veg_no_stem_ear_mask == 255)[0])
            veg2_necr = len(np.where(veg_col_mask_no_ear_no_stem[:, :, 1] == 61)[0])/len(np.where(veg_no_stem_ear_mask == 255)[0])

            # add color fractions to output data frame
            status_stat_names = ["ear_green", "ear_chlr", "ear_necr",
                                 "stem_green", "stem_chlr", "stem_necr",
                                 "veg_green", "veg_chlr", "veg_necr",
                                 "veg2_green", "veg2_chlr", "veg2_necr"]
            df[status_stat_names] = [[ear_green, ear_chlr, ear_necr,
                                      stem_green, stem_chlr, stem_necr,
                                      veg_green, veg_chlr, veg_necr,
                                      veg2_green, veg2_chlr, veg2_necr]]
            df.to_csv(self.path_stats / csv_name, index=False)
            result.put(file)

    def process_images(self):

        self.prepare_workspace()
        files = self.file_feed()

        if len(files) > 0:
            # make job and results queue
            m = Manager()
            jobs = m.Queue()
            results = m.Queue()
            processes = []
            # Progress bar counter
            max_jobs = len(files)
            count = 0

            # Build up job queue
            for file in files:
                print(file, "to queue")
                job = dict()
                job['file'] = file
                jobs.put(job)

            # Start processes
            for w in range(self.n_cpus):
                p = Process(target=self.process_image,
                            args=(jobs, results))
                p.daemon = True
                p.start()
                processes.append(p)
                jobs.put('STOP')

            print(str(len(files)) + " jobs started, " + str(self.n_cpus) + " workers")

            # Get results and increment counter along with it
            while count < max_jobs:
                img_names = results.get()
                count += 1
                print("processed " + str(count) + "/" + str(max_jobs))

            for p in processes:
                p.join()
