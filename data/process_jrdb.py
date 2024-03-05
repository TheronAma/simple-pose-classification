import os
import json
import numpy as np
import argparse
import ipdb
from constants import ACTIONS, TRAIN
from erica_split import get_jackrabbot_split



if __name__=="__main__":
    parser = argparse.ArgumentParser(
            prog = "Jackrabbot Pose Classification Preprocessor",
            description = """Preprocessor for jrdb poses in action
                             classification.""")
    parser.add_argument("-i", "--data_root", default="jackrabbot")
    parser.add_argument("-o", "--data_out", default="datasets/jackrabbot")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    TRAIN, VAL, TEST = get_jackrabbot_split()

    splits = { "train" : TRAIN, "val" : VAL}

    if args.verbose:
        print("loading poses 2d")
    poses_2d = np.load("poses_2d.npz", allow_pickle=True)['arr_0'].item()
    print(poses_2d.keys())

    poses = poses_2d['poses']
    for split in splits:
        split_data = []
        for scene in splits[split]:
            scene_poses = poses[scene]

            f = open(f"{args.data_root}/train/labels/labels_2d_stitched/{scene}.json")
            labels = json.load(f)["labels"]

            for frame in scene_poses:
                frame_poses = scene_poses[frame]
                frame_labels = labels[f'{frame:06d}.jpg']

                for label in frame_labels:
                    ped_id = int(label["label_id"].split(":")[1])
                    pose = frame_poses.get(ped_id, None)
                    if type(pose) is np.ndarray:
                        actions = label["action_label"]
                        for action in actions:
                            if actions[action] != 0 and action in ACTIONS:
                                pose_flattened = np.reshape(pose, (-1,))
                                data = np.concatenate([pose_flattened, np.array([ACTIONS.index(action)])], axis=0)
                                break
                        # print(data)
                        split_data.append(data)

        split_data = np.array(split_data)
        os.makedirs(f"{args.data_out}/{split}", exist_ok = True)
        np.save(f"{args.data_out}/{split}/data", split_data)
