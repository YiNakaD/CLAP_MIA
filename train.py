import os
import json
from datasets import Dataset, Audio
import torch
from transformers import AutoProcessor, ClapModel, Trainer, TrainingArguments, EarlyStoppingCallback
from models.model import Model
from config import set_random_seed
import datetime

now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d")
print(formatted_time)

set_random_seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # '0,1,2,3'
print(torch.cuda.device_count())

data_json_path = './data/train-clean-100.tar/temp/data.json'
audio_dir = './data/train-clean-100.tar/temp/'

# 加载json数据
with open(data_json_path, 'r') as f:
    reader_info = json.load(f)

# 构建数据集的列表,构造audio-text pair
data_samples = {
    'text': [],
    'audio': []
}
for reader_id, reader_data in reader_info.items():
    sex = 'female' if reader_data['sex'] == 'F' else 'male'
    name = reader_data['name']

    for chapter_id, chapter_info in reader_data.items():
        if isinstance(chapter_info, dict):
            book = chapter_info['book']
            chapter = chapter_info['title']

            # 构建identity_text
            identity_text = f"This audio is from {name}, a {sex} speaker of {book}, {chapter}."
            # identity_text = f" {name}, a {sex} speaker of {book}, {chapter}."

            # 查找音频文件的路径
            audio_path = os.path.join(audio_dir, reader_id, chapter_id)
            if os.path.exists(audio_path):
                for audio_file in os.listdir(audio_path):
                    if audio_file.endswith('.flac'):
                        audio_file_path = os.path.join(audio_path, audio_file)

                        # 将identity_text和音频路径加入样本
                        data_samples['text'].append(identity_text)
                        data_samples['audio'].append(audio_file_path)

# 使用datasets库将数据转换为Dataset
dataset = Dataset.from_dict(data_samples)
dataset = dataset.cast_column("audio", Audio())

train_test_split = dataset.train_test_split(test_size=0.1, shuffle=True)

# 获取训练集和测试集
# train_dataset = dataset
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']
print(len(train_dataset))

# 加载CLAP模型和处理器， Model是继承的ClapModel，在models/model.py里可以过去
model = Model.from_pretrained("laion/clap-htsat-unfused").to('cuda')  # 第一次训练的时候
# model = Model.from_pretrained("./checkpoints/model/checkpoint-1350-10-08-18-06").to('cuda')  # 使用已有模型的时候
processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")


# 定义处理函数，将数据转换为模型输入
def preprocess_function(batch):
    audio_samples = [item["audio"]["array"] for item in batch]  # 正确提取音频数组
    inputs = processor(text=[item['text'] for item in batch], audios=audio_samples, return_tensors="pt", padding=True,
                       sampling_rate=48000)
    inputs = {key: val.to('cpu') for key, val in inputs.items()}  # 将数据放入CPU
    return inputs


EPOCH = 300
LR = 3e-4

# 本地硬盘的挂载点
local_disk_mount_point = '/dev/***'
output_directory = f'{local_disk_mount_point}/vi/checkpoints/model'
log_directory = f'{local_disk_mount_point}/vi/checkpoints/log/epoch-{EPOCH}'
best_model_directory = f"{local_disk_mount_point}/vi/best_model_directory/epch-{EPOCH}-{formatted_time}"

# 使用 Trainer 进行训练
training_args = TrainingArguments(
    output_dir=output_directory,

    optim="adamw_torch",
    # weight_decay=0.01,
    learning_rate=LR,
    lr_scheduler_type='cosine',  # 使用余弦退火调度器
    warmup_ratio=0.1,  # 训练开始时10%的时间用于预热

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,

    num_train_epochs=EPOCH,

    logging_dir=log_directory,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    eval_strategy="epoch",
    eval_steps=10,

    load_best_model_at_end=True,  # 训练结束后加载最佳模型
    metric_for_best_model="eval_loss",  # "accuracy",  # 用于确定最佳模型的指标
    greater_is_better=False,
    report_to="wandb",

    save_total_limit=2,
    remove_unused_columns=False,
    dataloader_drop_last=True,
    # fp16=True,
)

# 创建 EarlyStoppingCallback 实例
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)

# 创建 Trainer 实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    # compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],
    tokenizer=processor,
    data_collator=lambda data: preprocess_function(data)
)

# 开始训练
trainer.train()

trainer.save_model(best_model_directory)
