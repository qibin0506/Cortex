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
        file_path = f"{data_root_dir}{self.file_names[idx]}"

        # 下载当前文件
        if not os.path.exists(file_path):
            if TrainerTools().parallel.is_main_process:
                print(f"正在下载{file_path}")
                dataset_snapshot_download(
                    'qibin0506/cortex-train-data-v2',
                    allow_file_pattern=[self.file_names[idx]],
                    local_dir=data_root_dir
                )

            TrainerTools().parallel.wait()

        # 删除并下载后一个文件
        if idx < len(self.file_names) - 1 and TrainerTools().parallel.is_main_process:
            next_file = self.file_names[idx + 1]
            dst_file = f'{data_root_dir}{next_file}'
            if os.path.exists(dst_file):
                os.remove(dst_file)

            threading.Thread(
                target=dataset_snapshot_download,
                kwargs={
                    'dataset_id': 'qibin0506/cortex-train-data-v2',
                    'allow_file_pattern': [next_file],
                    'local_dir': data_root_dir
                }
            ).start()

        # 删除前一个文件
        if idx > 0 and TrainerTools().parallel.is_main_process:
            prev_file = self.file_names[idx - 1]
            if os.path.exists(f'{data_root_dir}{prev_file}'):
                os.remove(f'{data_root_dir}{prev_file}')

            # with open('./data/pretrained.txt', 'a') as f:
            #     f.write(f'{prev_file},')

        return file_path


class PretrainStage0FileDataset(FileDatasetBase):
    def __init__(self):
        super().__init__([
            'mobvoi_seq_monkey_short_0.pkl',
            'mobvoi_seq_monkey_short_1.pkl',
            'mobvoi_seq_monkey_short_2.pkl',
            'mobvoi_seq_monkey_short_3.pkl',
            'mobvoi_seq_monkey_short_4.pkl',
            'mobvoi_seq_monkey_short_5.pkl',
            'mobvoi_seq_monkey_short_6.pkl',
            'mobvoi_seq_monkey_short_7.pkl',
            'mobvoi_seq_monkey_short_8.pkl',
            'wikipedia.pkl',
        ])


class PretrainStage1FileDataset(FileDatasetBase):
    def __init__(self):
        super().__init__([
            'mobvoi_seq_monkey_long_0.pkl',
            'mobvoi_seq_monkey_long_1.pkl'
        ])


class COTFileDataset(FileDatasetBase):
    def __init__(self):
        super().__init__(['cot_sft.pkl'])


class GRPOFileDataset(FileDatasetBase):
    def __init__(self):
        super().__init__(['grpo.pkl'])


class MixFileDataset(FileDatasetBase):
    def __init__(self):
        super().__init__(['mix_sft.pkl'])


class DPOFileDataset(FileDatasetBase):
    def __init__(self):
        super().__init__(['dpo.pkl'])


class DistillDataset(FileDatasetBase):
    def __init__(self):
        super().__init__([
            'cot_sft.pkl',
            'mix_sft.pkl'
        ])
