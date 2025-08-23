import os
import threading
from llm_trainer import FileDataset, TrainerTools
from constant import data_root_dir
import requests

from concurrent.futures import ThreadPoolExecutor
import  time


class NoOpPbar:
    """
    一个什么都不做的“哑”进度条类，用于在禁用tqdm时作为替代品。
    它实现了tqdm对象所需的基本方法和上下文管理协议。
    """

    def __init__(self, *args, **kwargs):
        pass

    def update(self, n=1):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MultiThreadDownloader:
    """
    一个用于多线程下载文件的类，
    增加了失败重试策略和可选的tqdm进度条。
    """

    def __init__(self, url, file_path, num_threads=8, max_retries=5, use_tqdm=True):
        """
        初始化下载器。

        :param url: 文件的 URL
        :param num_threads: 下载使用的线程数
        :param max_retries: 单个块下载失败后的最大重试次数
        :param use_tqdm: 是否使用tqdm显示进度条
        """
        self.url = url
        self.num_threads = num_threads
        self.max_retries = max_retries
        self.use_tqdm = use_tqdm  # 新增：控制是否使用tqdm
        self.file_size = 0
        self.file_path = file_path
        self.tqdm = None

        # 动态导入tqdm
        if self.use_tqdm:
            try:
                from tqdm import tqdm
                self.tqdm = tqdm
            except ImportError:
                self.use_tqdm = False

    def _get_file_info(self):
        # ... 此方法无改动 ...
        try:
            headers = {'Accept-Encoding': 'identity'}
            response = requests.head(self.url, headers=headers, allow_redirects=True, timeout=10)
            response.raise_for_status()
            if response.headers.get('Accept-Ranges') != 'bytes':
                print("服务器不支持多线程下载。将使用单线程下载。")
                self.num_threads = 1
            self.file_size = int(response.headers.get('Content-Length', 0))
            if self.file_size == 0:
                raise ValueError("无法获取文件大小或文件大小为0。")
            print(f"文件大小: {self.file_size / 1024 / 1024:.2f} MB")
            return True
        except requests.exceptions.RequestException as e:
            print(f"获取文件信息失败: {e}")
            return False

    def _download_chunk(self, start, end, pbar):
        # ... 此方法无改动 ...
        headers = {'Range': f'bytes={start}-{end}'}
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.url, headers=headers, stream=True, timeout=20)
                response.raise_for_status()
                bytes_downloaded_before_write = 0
                with open(self.file_path, 'rb+') as f:
                    f.seek(start)
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            bytes_downloaded_before_write += len(chunk)
                            f.write(chunk)
                            pbar.update(len(chunk))
                return
            except requests.exceptions.RequestException as e:
                if pbar: pbar.update(-bytes_downloaded_before_write)
                wait_time = 2 ** attempt
                # 只有在非静默模式下才打印重试信息
                if self.use_tqdm:
                    print(f"\n下载块 {start}-{end} 第 {attempt + 1} 次失败: {e}。将在 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        raise IOError(f"下载块 {start}-{end} 失败，已达到最大重试次数 {self.max_retries}。")

    def download(self):
        """
        执行多线程下载。
        """
        if not self._get_file_info():
            return

        with open(self.file_path, 'wb') as f:
            f.seek(self.file_size - 1)
            f.write(b'\0')

        chunk_size = self.file_size // self.num_threads
        ranges = []
        for i in range(self.num_threads):
            start = i * chunk_size
            end = start + chunk_size - 1 if i < self.num_threads - 1 else self.file_size - 1
            ranges.append((start, end))

        # 根据配置选择进度条类
        pbar_class = self.tqdm if self.use_tqdm else NoOpPbar

        try:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                with pbar_class(total=self.file_size, unit='B', unit_scale=True, desc=self.file_path) as pbar:
                    futures = [executor.submit(self._download_chunk, start, end, pbar) for start, end in ranges]
                    for future in futures:
                        future.result()

            print(f"\n文件 '{self.file_path}' 下载完成！")
        except Exception as e:
            print(f"\n下载过程中发生致命错误: {e}")
            print("下载失败，请检查网络或文件链接后重试。")
            # 可选：删除不完整的文件
            # os.remove(self.file_name)


def _download(url, file_path):
    num_threads = 8 # 1
    MultiThreadDownloader(url=url, file_path=file_path, num_threads=num_threads).download()


class FileDatasetBase(FileDataset):

    def __init__(self, file_names: list):
        self.file_names = file_names
        self.download_url = 'https://www.modelscope.cn/datasets/qibin0506/cortex-train-data-v2/resolve/master/{}'

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx) -> str:
        file_path = f"{data_root_dir}{self.file_names[idx]}"

        # 下载当前文件
        if not os.path.exists(file_path):
            if TrainerTools().parallel.is_main_process:
                print(f"正在下载{file_path}")
                _download(self.download_url.format(self.file_names[idx]), file_path)
                # _download(self.download_url.format(self.file_names[idx]), file_path)

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