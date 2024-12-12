import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from transformers import AutoProcessor
import csv
from datasets import Dataset, Audio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from tqdm import tqdm
from models.model import Model
import warnings
import datetime, json
from config import set_random_seed
from pathlib import Path
import copy
import time

now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d")
print(formatted_time)

set_random_seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print("torch.cuda.device_count %d" % torch.cuda.device_count())

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

print("len(data_samples['text']) = %d" % len(data_samples['text']))


# 读取文件列表
def get_random_audio_path(audio_dir):
    audio_files = os.listdir(audio_dir)
    audio_files = [file for file in audio_files if file.endswith('.flac')]
    if not audio_files:
        raise ValueError(f"No audio files found in directory {audio_dir}")
    return os.path.join(audio_dir, random.choice(audio_files))


# 处理音频
def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 48000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=48000)(waveform)
    duration = waveform.size(1) / 48000
    print(f"Original audio duration: {duration:.2f} seconds")
    num_samples = int(duration * 48000)
    random_waveform = np.random.uniform(-1, 1, num_samples).astype(np.float32)
    return random_waveform


# 获取features
def extract_features(audio, text_description, processor, model):
    inputs = processor(text=text_description, audios=audio, return_tensors="pt", padding=True, sampling_rate=48000)
    output = model(**inputs)
    return output.text_embeds, output.audio_embeds


# local_disk_mount_point = '/dev/shm'
# clap_model_path = f"{local_disk_mount_point}/vi/best_model_directory/epch-300-2024-10-16"
# clap_model_path = "./checkpoints/model/checkpoint-epoch-90-batchsize-64-10-08-19-030"
clap_model_path = "./checkpoints/model/epch-300-2024-10-10"

EPOCH = 100

processor = AutoProcessor.from_pretrained(clap_model_path)
model = Model.from_pretrained(clap_model_path).to("cuda")
model = torch.nn.DataParallel(model)


# 优化部分
def main(audio_path, text_description):
    audio = preprocess_audio(audio_path)
    audio_tensor = torch.tensor(audio, requires_grad=True)
    audio_tensor.requires_grad = True
    print('audio: ', audio_tensor)
    optimizer = optim.AdamW([audio_tensor], lr=3e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    min_loss = 100.0
    max_loss = -1
    best_audio_features = None

    index = 0
    final_text_features, best_audio_features = extract_features(audio_tensor.detach().cpu().numpy(), text_description,
                                                                processor, model)

    for iteration in range(EPOCH):
        optimizer.zero_grad()

        # 提取特征
        text_features, audio_features = extract_features(audio_tensor.detach().cpu().numpy(), text_description,
                                                         processor, model)

        loss_tensor = torch.tensor(1) - nn.functional.cosine_similarity(audio_features, text_features)
        loss_tensor.backward()
        optimizer.step()
        scheduler.step()
        index += 1

        if min_loss > loss_tensor.item():
            min_loss = loss_tensor.item()
            best_audio_features = audio_features.clone()
            best_index = copy.deepcopy(index)

        if max_loss < loss_tensor.item():
            max_loss = loss_tensor.item()

        # print(f"Iteration {iteration + 1}, Loss: {loss_tensor.item()}, Learning Rate: {scheduler.get_last_lr()[0]}")

    final_similarity = 1 - nn.functional.cosine_similarity(best_audio_features, final_text_features)
    print(f"Final 1-Similarity: {final_similarity.item()},best_index:{best_index}")

    # 保存特征向量
    # save_path_features = f'./data/features/{text_description.replace(" ", "_")}_features.csv'
    if max_loss != min_loss:
        file_name = audio_path.split('\\')[-1][:-5]
        save_path_features = f'./data/features/{file_name}_features.csv'
        np.savetxt(save_path_features, best_audio_features.detach().cpu().numpy(), delimiter=',')
        print(f"Feature vectors for {text_description} saved to {save_path_features}")

    print('audio: ', audio_tensor)


if __name__ == "__main__":
    folder_path = Path('./data/features')
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    # 使用 zip() 将两个列表配对
    zipped = zip(data_samples['audio'], data_samples['text'])

    shuffle_data_samples = list(zipped)
    random.shuffle(shuffle_data_samples)

    start_time = time.time()

    num = 0
    for sample in tqdm(shuffle_data_samples):
        num += 1
        audio_path = sample[0]
        text_description = sample[1]
        main(audio_path, text_description)

        if num >= 9:
            break

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f'运行时间: {elapsed_time:.4f} 秒')  # 输出运行时间

