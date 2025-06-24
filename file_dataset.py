import threading
from llm_trainer import FileDataset, TrainerTools
from constant import data_root_dir
import requests, os

def log(msg):
    print(msg)
    with open('log.txt', 'a') as f:
        f.write(f"{msg}\n")


def _download(url: str, dst: str, chunk_size=1024 * 1024, retry_count=0):
    if retry_count > 2 or os.path.exists(dst):
        return

    log(f'_download: {dst}')

    try:
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            with open(dst, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        if not isinstance(e, KeyboardInterrupt):
            _download(url, dst, chunk_size, retry_count=retry_count + 1)


class FileDatasetBase(FileDataset):

    def __init__(self, file_names: list):
        self.file_names = file_names
        self.download_url = 'https://www.modelscope.cn/datasets/qibin0506/cortex-train-data-1024/resolve/master/data/{}'

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx) -> str:
        file_path = f"{data_root_dir}{self.file_names[idx]}"

        # 下载当前文件
        if not os.path.exists(file_path):
            if TrainerTools().parallel.is_main_process:
                print(f"正在下载{file_path}")
                _download(self.download_url.format(self.file_names[idx]), file_path)

            TrainerTools().parallel.wait()

        # 删除并下载后一个文件
        if idx < len(self.file_names) - 1 and TrainerTools().parallel.is_main_process:
            next_file = self.file_names[idx + 1]
            dst_file = f'{data_root_dir}{next_file}'
            if os.path.exists(dst_file):
                os.remove(dst_file)

            threading.Thread(target=_download, args=(self.download_url.format(next_file), dst_file)).start()

        # 删除前一个文件
        if idx > 0 and TrainerTools().parallel.is_main_process:
            prev_file = self.file_names[idx - 1]
            if os.path.exists(f'{data_root_dir}{prev_file}'):
                os.remove(f'{data_root_dir}{prev_file}')

            # with open('./data/pretrained.txt', 'a') as f:
            #     f.write(f'{prev_file},')

        return file_path


class PretrainFileDataset(FileDatasetBase):
    def __init__(self):
        super().__init__([
            'pretrain_long_0.pkl',
            'pretrain_long_1.pkl',
            'pretrain_long_2.pkl',
            'pretrain_long_3.pkl',
            'pretrain_long_4.pkl',
            'pretrain_long_5.pkl',
            'pretrain_long_6.pkl',
            'pretrain_long_7.pkl',
            'pretrain_long_8.pkl',
            'pretrain_long_final.pkl',
            'pretrain_short_0.pkl',
            'pretrain_short_1.pkl',
            'pretrain_short_2.pkl',
            'pretrain_short_3.pkl',
            'pretrain_short_final.pkl',
        ])


class SFTFileDataset(FileDatasetBase):
    def __init__(self):
        super().__init__(['sft.pkl'])


class ReasoningFileDataset(FileDatasetBase):
    def __init__(self):
        super().__init__(['reasoning_data.pkl'])


class DPOFileDataset(FileDatasetBase):
    def __init__(self):
        super().__init__(['dpo.pkl'])


class GRPOFileDataset(FileDatasetBase):
    def __init__(self):
        super().__init__(['grpo.pkl'])
