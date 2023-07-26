"""
utils.py
"""

import os
import shutil
from typing import Any, Optional

import yaml
from loguru import logger
from pycocotools.coco import COCO


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
    ) -> None:
        self._coco = None
        self._coco_ann_path = None
        self._output_dir = None
        self._coco_imgs_dir = None
        self._yolo_labels_dir = None
        self._yolo_images_dir = None
        self._yolo_cfg_yaml_path = None
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
        if (
            ann_path is not None
            and os.path.exists(ann_path)
            and os.path.isfile(ann_path)
        ):
            self._coco_ann_path = ann_path
            self._coco = COCO(self._coco_ann_path)

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
            if os.path.exists(cfg_yaml_path):
                logger.warning(f"Path exist, be overwritten:{cfg_yaml_path}")
            self._yolo_cfg_yaml_path = cfg_yaml_path
        else:
            self._yolo_cfg_yaml_path = os.path.join(self.output_dir, "yolo.yml")

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
        elif os.path.isdir(output_dir):
            self._output_dir = output_dir
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

    def write_labels(self):
        """
        write labels to files
        """
        if not os.path.exists(self._yolo_labels_dir):
            os.mkdir(self._yolo_labels_dir)

        for img_id in self._coco.getImgIds():
            fname = (
                os.path.splitext(self._coco.imgs[img_id]["file_name"])[0]
                + ".txt"
            )
            with open(
                os.path.join(self._yolo_labels_dir, fname),
                "w",
                encoding="UTF-8",
            ) as f_handler:
                f_handler.write(self._img_to_labels(img_id=img_id))

    def write_images(self):
        """
        write images to yolo datasets
        """
        if self.coco_imgs_dir is None or not os.path.exists(self.coco_imgs_dir):
            logger.error("images directory path not set")
            return

        if not os.path.exists(self._yolo_images_dir):
            os.mkdir(self._yolo_images_dir)

        for img_id in self._coco.getImgIds():
            source_path = os.path.join(
                self._coco_imgs_dir, self._coco.imgs[img_id]["file_name"]
            )
            shutil.copy(source_path, self._yolo_images_dir)

    def write_yaml(self):
        """
        write yaml file
        """
        yaml_content_dict = {"path": "", "train": "", "test": None, "names": ""}
        yaml_content_dict["names"] = {
            int(self._coco.cats[cid]["id"]) - 1: self._coco.cats[cid]["name"]
            for cid in self._coco.getCatIds()
        }

        with open(self.yolo_cfg_yaml_path, "w", encoding="UTF-8") as f_handler:
            yaml.dump(yaml_content_dict, f_handler)

    def __call__(self) -> Any:
        if self.coco is None:
            logger.error("coco is None")
            return

        self.write_labels()
        self.write_images()
        self.write_yaml()
