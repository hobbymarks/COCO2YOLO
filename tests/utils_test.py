"""
utils test
"""
import importlib.resources as pkg_resources
import os
import shutil

from c2y import Xcoco

package = pkg_resources.files("assets")


def test_xcoco():
    """
    test xcoco init
    """
    xcoco = Xcoco()
    assert xcoco.coco_ann_path is None
    assert xcoco.output_dir is not None
    assert xcoco.coco is None
    assert xcoco.coco_imgs_dir is None
    assert xcoco.yolo_labels_dir is not None
    assert xcoco.yolo_images_dir is not None
    xcoco()

    xcoco = Xcoco(coco_ann_path=package / "edgeshots" / "annotations")
    assert xcoco.coco_ann_path is None
    assert xcoco.output_dir is not None
    assert xcoco.coco is None
    assert xcoco.coco_imgs_dir is None
    assert xcoco.yolo_labels_dir is not None
    assert xcoco.yolo_images_dir is not None
    xcoco()

    xcoco = Xcoco(
        coco_ann_path=package
        / "edgeshots"
        / "annotations"
        / "instances_default.json"
    )
    assert xcoco.coco_ann_path is not None
    assert xcoco.output_dir is not None
    assert xcoco.coco is not None
    assert xcoco.coco_imgs_dir is None
    assert xcoco.yolo_labels_dir is not None
    assert xcoco.yolo_images_dir is not None
    xcoco()
    assert os.path.exists(xcoco.yolo_labels_dir) is True
    if os.path.exists(xcoco.yolo_labels_dir):
        shutil.rmtree(xcoco.yolo_labels_dir)
    assert os.path.exists(xcoco.yolo_images_dir) is False
    if os.path.exists(xcoco.yolo_images_dir):
        shutil.rmtree(xcoco.yolo_images_dir)
    assert os.path.exists(xcoco.yolo_cfg_yaml_path) is True
    if os.path.exists(xcoco.yolo_cfg_yaml_path):
        os.remove(xcoco.yolo_cfg_yaml_path)

    xcoco = Xcoco(
        coco_ann_path=package
        / "edgeshots"
        / "annotations"
        / "instances_default.json",
        coco_imgs_dir=package / "edgeshots" / "images",
    )
    assert xcoco.coco_ann_path is not None
    assert xcoco.output_dir is not None
    assert xcoco.coco is not None
    assert xcoco.coco_imgs_dir is not None
    assert xcoco.yolo_labels_dir is not None
    assert xcoco.yolo_images_dir is not None
    xcoco()
    assert os.path.exists(xcoco.yolo_labels_dir) is True
    if os.path.exists(xcoco.yolo_labels_dir):
        shutil.rmtree(xcoco.yolo_labels_dir)
    assert os.path.exists(xcoco.yolo_images_dir) is True
    if os.path.exists(xcoco.yolo_images_dir):
        shutil.rmtree(xcoco.yolo_images_dir)
    assert os.path.exists(xcoco.yolo_cfg_yaml_path) is True
    if os.path.exists(xcoco.yolo_cfg_yaml_path):
        os.remove(xcoco.yolo_cfg_yaml_path)
