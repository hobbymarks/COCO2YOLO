"""
utils.py
"""

import os
import shutil
from typing import Any, Optional

import yaml
from pycocotools.coco import COCO


class Xcoco:
    """
    Xcoco
    """

    def __init__(
        self,
        coco_annotation_path: Optional[str] = None,
        coco_images_dir_path: Optional[str] = None,
        output_dir_path: Optional[str] = None,
    ) -> None:
        self._coco = None
        self.coco_annotaions_path = None
        self.out_dir = None
        self._coco_images_dir = None
        self._yolo_labels_dir = None
        self._yolo_images_dir = None
        self.set_coco_annotation_path(ann_path=coco_annotation_path)
        self.set_coco_images_dir_path(coco_images_dir_path=coco_images_dir_path)
        self.set_output_dir_path(output_dir_path=output_dir_path)

    def set_coco_annotation_path(self, ann_path: str):
        """
        set annoation file path
        """
        self.coco_annotaions_path = ann_path
        if os.path.exists(self.coco_annotaions_path):
            self._coco = COCO(self.coco_annotaions_path)

    def set_coco_images_dir_path(
        self,
        coco_images_dir_path: Optional[str] = None,
    ):
        """
        set images directory path
        """
        if coco_images_dir_path is not None and os.path.isdir(
            coco_images_dir_path
        ):
            self._coco_images_dir = coco_images_dir_path

    def set_output_dir_path(self, output_dir_path: Optional[str] = None):
        """
        set output directory path
        """
        if output_dir_path is None:
            self.out_dir = "./"
        elif os.path.isdir(output_dir_path):
            self.out_dir = output_dir_path
        else:
            raise Exception(f"{output_dir_path}")

        self._yolo_labels_dir = os.path.join(self.out_dir, "labels")
        self._yolo_images_dir = os.path.join(self.out_dir, "images")

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
            out_text += f"{ann['category_id']}"
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
        if not os.path.exists(self._yolo_images_dir):
            os.mkdir(self._yolo_images_dir)

        for img_id in self._coco.getImgIds():
            source_path = os.path.join(
                self._coco_images_dir, self._coco.imgs[img_id]["file_name"]
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

        with open(
            os.path.join(self.out_dir, "coco.yml"), "w", encoding="UTF-8"
        ) as f_handler:
            yaml.dump(yaml_content_dict, f_handler)

    def __call__(self) -> Any:
        self.write_labels()
        self.write_images()
        self.write_yaml()
