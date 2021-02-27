import os
import json
from typing import List, Dict
from collections import defaultdict


class ProjectIdsPreprocessor:
    """Class to process separate msgs & project ids files into one json."""
    @staticmethod
    def map_ids_to_msgs(ids: List[str], msgs: List[str]) -> Dict[str, str]:
        ids_to_msgs = defaultdict(list)
        for id, msg in zip(ids, msgs):
            id, msg = id.strip(), msg.strip()
            ids_to_msgs[id].append(msg + ' \n ')
        return ids_to_msgs

    @staticmethod
    def create_file(ds_root_path: str):
        cur_path = os.path.join(ds_root_path, 'train')
        with open(os.path.join(cur_path, 'projectIds.txt')) as proj_ids_file, \
                open(os.path.join(cur_path, 'msg.txt')) as msg_file, \
                open(os.path.join(cur_path, 'ids_to_msg.json'), 'w') as target_file:
            ids_to_msgs = ProjectIdsPreprocessor.map_ids_to_msgs(proj_ids_file.readlines(), msg_file.readlines())
            json.dump(ids_to_msgs, target_file)


if __name__ == '__main__':
    dataset_path = '../raw_data/CleanedJiang/'
    ProjectIdsPreprocessor.create_file(dataset_path)
