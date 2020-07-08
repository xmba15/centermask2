#!/usr/bin/env python

__all__ = ["get_json_object_from_json_all_labels", "parse_via_json_all_labels"]


def get_json_object_from_json_all_labels(json_path):
    import json

    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(e)
        raise


def parse_via_json_all_labels(json_object, classes, prefix_path=""):
    from detectron2.structures import BoxMode
    import os
    import cv2
    import numpy as np

    dataset_dicts = []
    for key, value in json_object.items():
        file_name = os.path.join(prefix_path, value["filename"])

        img = cv2.imread(file_name, 0)
        if img is None:
            raise Exception("Failed to open {}\n".format(file_name))

        height, width = img.shape[:2]

        record = {}
        record["file_name"] = file_name
        record["height"] = height
        record["width"] = width

        objs = []
        for region in value["regions"]:
            shape_attributes = region["shape_attributes"]
            region_attributes = region["region_attributes"]
            px = shape_attributes["all_points_x"]
            py = shape_attributes["all_points_y"]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            if not region_attributes:
                category_id = 0
            else:
                category_id = classes.index(region_attributes["label"])

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": category_id,
                "iscrowd": 0,
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def main(args):
    import random
    import cv2
    from detectron2.utils.visualizer import Visualizer

    json_object = get_json_object_from_json_all_labels(args.json_path)
    classes = [v.strip() for v in args.classes.split(",")]
    dataset_dicts = parse_via_json_all_labels(json_object, classes, args.prefix_path)

    from detectron2.data import MetadataCatalog

    metadata = MetadataCatalog.get("dataset_test").set(thing_classes=classes)

    for d in random.sample(dataset_dicts, 10):
        img = cv2.imread(d["file_name"])[:, :, ::-1]
        img_copy = img.copy()
        visualizer = Visualizer(img_copy, metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow("test_visualization", vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyWindow("test_visualization")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("test parsing json")
    parser.add_argument(
        "--json_path", type=str, required=True, help="path to label json"
    )
    parser.add_argument("--classes", type=str, required=True, help="path to label json")
    parser.add_argument(
        "--prefix_path", type=str, default="", help="absolute path that holds images"
    )
    args = parser.parse_args()

    main(args)
