import noisereduce as nr
import librosa
import os
import soundfile as sf  # 导入 soundfile 库

# 降噪函数
def denoise_audio(input_folder, output_folder):
    # 计数器
    processed_count = 0
    
    for filename in os.listdir(input_folder):
        # 检查是否是音频文件（这里只处理 .wav 格式的文件）
        if filename.endswith('.wav'):
            file_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # 加载音频文件
            audio, sr = librosa.load(file_path, sr=None)
            
            # 使用 noisereduce 降噪
            reduced_noise_audio = nr.reduce_noise(y=audio, sr=sr)
            
            # 保存降噪后的音频
            sf.write(output_path, reduced_noise_audio, sr)
            
            # 更新计数器
            processed_count += 1
            
            # 每处理100个文件输出信息
            if processed_count % 100 == 0:
                print(f"已处理 {processed_count} 个文件: {input_folder}")

# # 对每个文件夹中的音频文件进行降噪
# for folder in input_dirs:
#     print(f"开始处理 {folder} 文件夹中的音频...")
#     denoise_audio(input_dirs[folder], output_dirs[folder])
#     print(f"{folder} 文件夹处理完成！")

# print("所有音频降噪处理完成！")

# 使用示例：对 train、dev 和 test 文件夹中的所有 WAV 文件进行降噪
train_folder = "./train/"
dev_folder = "./dev/"
test_folder = "./test/"
output_train_folder = "./denoised_train/"
output_dev_folder = "./denoised_dev/"
output_test_folder = "./denoised_test/"

# 对 train、dev 和 test 文件夹中的音频文件进行降噪
denoise_audio(train_folder, output_train_folder)
denoise_audio(dev_folder, output_dev_folder)
denoise_audio(test_folder, output_test_folder)
