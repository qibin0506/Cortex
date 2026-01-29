import os
import threading
from llm_trainer import FileDataset, TrainerTools
from constant import data_root_dir
from modelscope import dataset_snapshot_download


class FileDatasetBase(FileDataset):

    def __init__(self, file_names: list):
        self.file_names = file_names

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx) -> str:
        file_path = f"{data_root_dir()}{self.file_names[idx]}"

        # 下载当前文件
        if not os.path.exists(file_path):
            if TrainerTools().parallel.is_main_process:
                dataset_snapshot_download(
                    'qibin0506/Cortex-3.0-data',
                    allow_file_pattern=[self.file_names[idx]],
                    local_dir=data_root_dir()
                )

            TrainerTools().parallel.wait()

        # 下载后一个文件
        if idx < len(self.file_names) - 1 and TrainerTools().parallel.is_main_process:
            next_file = self.file_names[idx + 1]
            dst_file = f'{data_root_dir()}{next_file}'
            if not os.path.exists(dst_file):
                threading.Thread(
                    target=dataset_snapshot_download,
                    kwargs={
                        'dataset_id': 'qibin0506/Cortex-3.0-data',
                        'allow_file_pattern': [next_file],
                        'local_dir': data_root_dir()
                    }
                ).start()

        # 删除前一个文件
        if idx > 0 and TrainerTools().parallel.is_main_process:
            prev_file = self.file_names[idx - 1]
            if os.path.exists(f'{data_root_dir()}{prev_file}'):
                os.remove(f'{data_root_dir()}{prev_file}')

        return file_path


class PretrainFileDataset(FileDatasetBase):
    def __init__(self):
        super().__init__([
            'pretrain_data_0.npy',
            'pretrain_data_1.npy',
        ])


class MidtrainFileDataset(FileDatasetBase):
    def __init__(self):
        super().__init__([
            'midtrain_data_0.npy',
        ])


class SFTFileDataset(FileDatasetBase):
    def __init__(self):
        super().__init__([
            'sft_data.npy',
        ])


class PPODataset(FileDatasetBase):
    def __init__(self):
        super().__init__(['ppo_data.npy'])
