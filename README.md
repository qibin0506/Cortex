# Cortex
个人从零训练一个MoE LLM，训练过程包括Pretrain、SFT、Reasoning、DPO、GRPO

---

<div align="center">
  <img src="./images/screenshot.png">
</div>

在线体验：[https://s.c1ns.cn/cortex](https://s.c1ns.cn/cortex)

*在线体验服务不稳定，如果不能访问，可自己本地部署体验*

**模型尺寸：0.6B，MoE推理激活参数0.2B**

本项目提供训练各个阶段checkpoint，可按需下载。下载地址：[https://www.modelscope.cn/models/qibin0506/Cortex](https://www.modelscope.cn/models/qibin0506/Cortex/files)

---

### 本机部署
1. 确保本机已安装python3
2. clone或下载本项目
3. 安装依赖 `pip3 install -r requirements.txt`
4. 下载checkpoint [last_checkpoint.bin](https://www.modelscope.cn/models/qibin0506/Cortex/resolve/master/last_checkpoint.bin)，并放置到项目根目录
5. 执行 `python3 app.py`运行项目，访问链接[http://0.0.0.0:8080/](http://0.0.0.0:8080/) 即可体验

### 自己训练
安装上面本机部署部分安装依赖，如果需要继续训练，可按需下载checkpoint，然后进行训练，训练方法如下：
1. 预训练：`smart_train train_pretrain.py`
2. SFT：`smart_train train_sft.py
3. 推理能力：`smart_train train_reasoning.py`
4. DPO：`smart_train train_dpo.py`
5. GRPO：`smart_train train_grpo.py`

注意：
1. 参数可在utils.py文件中修改，内置参数均是4*4090设备上使用的参数
2. 本项目会自动管理训练文件，无需提前下载、组织训练数据
