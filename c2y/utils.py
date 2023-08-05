"""
utils.py
"""

import io
import os
import shutil
import tempfile
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Optional

import magic
import oyaml as yaml
from loguru import logger
from pycocotools.coco import COCO

COCO_ANN_DIR = "annotations"
COCO_IMGS_DIR = "images"


def capture_logger(func):
    """
    capture stdout and stderr to log
    """

    def wrapper(*args, **kwargs):
        # Create file-like objects
        stdout_stream = io.StringIO()
        stderr_stream = io.StringIO()

        # Redirect stdout and stderr
        with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
            # Call the function
            func(*args, **kwargs)

        # Log output
        if (_stdout := stdout_stream.getvalue()) != "":
            logger.trace(f"{_stdout}")
        if (_stderr := stderr_stream.getvalue()) != "":
            logger.trace(f"{_stderr}")

    return wrapper


class Xcoco:
    """
    Xcoco
    """

    def __init__(
        self,
        ann_path: Optional[str] = None,
        imgs_dir: Optional[str] = None,
        yolo_cfg_yaml: Optional[str] = None,
        output_dir: Optional[str] = None,
        force: bool = False,
    ) -> None:
        self._coco = None
        self._coco_ann_path = None
        self._output_dir = None
        self._coco_imgs_dir = None
        self._yolo_labels_dir = None
        self._yolo_images_dir = None
        self._yolo_cfg_yaml_path = None
        self._force = force
        self.coco_ann_path = ann_path
        self.coco_imgs_dir = imgs_dir
        self.output_dir = output_dir
        self.yolo_cfg_yaml_path = yolo_cfg_yaml

    @property
    def coco(self):
        """
        coco getter
        """
        return self._coco

    @property
    def coco_ann_path(self):
        """
        coco annotaions path getter
        """
        return self._coco_ann_path

    @coco_ann_path.setter
    def coco_ann_path(self, ann_path: str):
        """
        set annoation file path
        """

        def _mimetype(fpath):
            m_obj = magic.Magic(mime=True)
            return m_obj.from_file(fpath)

        def _ann_name(files):
            _anns = [f for f in files if f.startswith(COCO_ANN_DIR)]
            if len(_anns) == 1:
                return _anns[0]
            return None

        def _imgs_name(files):
            _imgs = [f for f in files if f.startswith(COCO_IMGS_DIR)]
            if len(_imgs) > 0:
                return _imgs
            return None

        @capture_logger
        def _load_coco(ann_path: str):
            self._coco_ann_path = ann_path
            logger.trace(f"Set COCO ann path:{self._coco_ann_path}")
            self._coco = COCO(self._coco_ann_path)

        if (
            ann_path is not None
            and os.path.exists(ann_path)
            and os.path.isfile(ann_path)
        ):
            _mime = str(_mimetype(ann_path))
            if _mime == "application/zip":  # if zip
                _temp_dir = tempfile.gettempdir()
                with zipfile.ZipFile(ann_path, "r") as zip_ref:
                    _z_names = zip_ref.namelist()
                    if (_ann := _ann_name(_z_names)) is not None:
                        zip_ref.extract(
                            _ann, path=_temp_dir
                        )  # set coco annotations path
                        _load_coco(os.path.join(_temp_dir, _ann))
                        if (self.coco_imgs_dir is None) and (
                            (_imgs := _imgs_name(_z_names)) is not None
                        ):  # set coco images directory
                            for f_name in _imgs:
                                zip_ref.extract(f_name, _temp_dir)
                            self.coco_imgs_dir = os.path.join(
                                _temp_dir, COCO_IMGS_DIR
                            )
            elif _mime == "application/json":  # if json
                _load_coco(ann_path)

    @property
    def coco_imgs_dir(self):
        """
        coco images directory path getter
        """
        return self._coco_imgs_dir

    @coco_imgs_dir.setter
    def coco_imgs_dir(
        self,
        coco_imgs_dir: Optional[str] = None,
    ):
        """
        set images directory path
        """
        if coco_imgs_dir is not None and os.path.isdir(coco_imgs_dir):
            self._coco_imgs_dir = coco_imgs_dir
            logger.trace(f"Set COCO images dir:{self._coco_imgs_dir}")

    @property
    def yolo_cfg_yaml_path(self):
        """
        yolo yaml config path getter
        """
        return self._yolo_cfg_yaml_path

    @yolo_cfg_yaml_path.setter
    def yolo_cfg_yaml_path(
        self,
        cfg_yaml_path: Optional[str] = None,
    ):
        """
        set yolo yaml config path
        """
        if cfg_yaml_path is not None:
            self._yolo_cfg_yaml_path = cfg_yaml_path
        else:
            self._yolo_cfg_yaml_path = os.path.join(
                self.output_dir, "yolo.yml"
            )
        logger.trace(f"Set YOLO config yaml path:{self._yolo_cfg_yaml_path}")

    @property
    def output_dir(self):
        """
        output directory path getter
        """
        return self._output_dir

    @output_dir.setter
    def output_dir(self, output_dir: Optional[str] = None):
        """
        set output directory path
        """
        if output_dir is None:
            self._output_dir = "./"
            logger.trace(f"Set Output dir:{self._output_dir}")
        elif os.path.isdir(output_dir):
            self._output_dir = output_dir
            logger.trace(f"Set Output dir:{self._output_dir}")
        else:
            raise Exception(f"{output_dir}")

        self._yolo_labels_dir = os.path.join(self._output_dir, "labels")
        self._yolo_images_dir = os.path.join(self._output_dir, "images")

    @property
    def yolo_labels_dir(self):
        """
        yolo labels directory path getter
        """
        return self._yolo_labels_dir

    @property
    def yolo_images_dir(self):
        """
        yolo images directory path getter
        """
        return self._yolo_images_dir

    def _bbox_2_yolo(self, bbox, img_w, img_h):
        loc_x, loc_y, bx_w, bx_h = bbox[0], bbox[1], bbox[2], bbox[3]
        center_x = loc_x + bx_w / 2
        center_y = loc_y + bx_h / 2

        center_x *= 1 / img_w
        bx_w *= 1 / img_w
        center_y *= 1 / img_h
        bx_h *= 1 / img_h
        return center_x, center_y, bx_w, bx_h

    def _img_to_labels(self, img_id):
        """
        xx
        """
        img_w, img_h = (
            self._coco.imgs[img_id]["width"],
            self._coco.imgs[img_id]["height"],
        )

        out_text = ""
        for ann in self._coco.imgToAnns[img_id]:
            _rlt = self._bbox_2_yolo(ann["bbox"], img_w, img_h)
            out_text += f"{int(ann['category_id'])-1}"
            for _r in _rlt:
                out_text += f" {_r:.6f}"
            out_text += "\n"

        return out_text

    def _prepare_dataset_dir(self, dir_path) -> bool:
        if not os.path.exists(dir_path):  # not exist
            os.mkdir(dir_path)
            return True
        # exist
        if not Path(dir_path).is_dir():  # exist but not dir
            logger.error(f"Not Directory:{dir_path}")
            return False

        if self._force is True:  # exist and is dir and force is True
            logger.trace(f"force is {self._force}")
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
            return True
        # exist and is dir and force is False
        _in = input(f"{dir_path} exist.\nare you surce to overwrite?yes/(no)")
        if _in.lower() == "y" or _in.lower() == "yes":  # input yes
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
            return True

        return False

    def write_labels(self):
        """
        write labels to files
        """
        if not self._prepare_dataset_dir(self._yolo_labels_dir):
            return

        for img_id in self._coco.getImgIds():
            fname = (
                os.path.splitext(self._coco.imgs[img_id]["file_name"])[0]
                + ".txt"
            )
            _txt = self._img_to_labels(img_id=img_id)
            if _txt == "":
                continue
            _f_label = os.path.join(self._yolo_labels_dir, fname)
            with open(
                _f_label,
                "w",
                encoding="UTF-8",
            ) as f_handler:
                f_handler.write(_txt)
                logger.trace(f"Writle label file:{_f_label}")

    def write_images(self):
        """
        write images to yolo datasets
        """
        if self.coco_imgs_dir is None or not os.path.exists(
            self.coco_imgs_dir
        ):
            logger.error("coco images directory not set")
            return

        if not self._prepare_dataset_dir(self._yolo_images_dir):
            return

        for img_id in self._coco.getImgIds():
            source_path = os.path.join(
                self._coco_imgs_dir, self._coco.imgs[img_id]["file_name"]
            )
            shutil.copy(source_path, self._yolo_images_dir)
            logger.trace(f"Copy image file:{source_path}")

    def write_yaml(self):
        """
        write yaml file
        """
        if os.path.exists(self.yolo_cfg_yaml_path):
            if self._force is not True:
                _in = input(
                    f"{self.yolo_cfg_yaml_path} exist.\nare you sure to overwrite? yes/(no)"
                )
                if (
                    _in.lower() != "y" and _in.lower() != "yes"
                ):  # input not yes
                    return

            with open(
                self.yolo_cfg_yaml_path, "r", encoding="UTF-8"
            ) as f_handler:
                yaml_content_dict = yaml.load(
                    f_handler, Loader=yaml.FullLoader
                )
        else:
            yaml_content_dict = {
                "path": "",
                "train": "",
                "test": None,
                "names": "",
            }

        yaml_content_dict["names"] = {
            int(self._coco.cats[cid]["id"]) - 1: self._coco.cats[cid]["name"]
            for cid in self._coco.getCatIds()
        }

        with open(self.yolo_cfg_yaml_path, "w", encoding="UTF-8") as f_handler:
            yaml.dump(yaml_content_dict, f_handler)
            logger.trace(f"Write yolo config file:{self.yolo_cfg_yaml_path}")

    def __call__(self) -> Any:
        if self.coco is None:
            logger.error("coco is None")
            return

        self.write_labels()
        self.write_images()
        self.write_yaml()
