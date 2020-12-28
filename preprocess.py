import os
import librosa
import json

#dataset and mfcc path settings
DATASET_PATH = os.path.join(os.getcwd(), 'audio')
MFCC_PATH = os.path.join(os.getcwd(), 'mfcc')
print(DATASET_PATH, MFCC_PATH)
#mfcc parameters
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 22050

#extract mfcc feature from audio file
def extract_mfcc(audio_path, sr=SAMPLE_RATE, num_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):
    signal, sample_rate = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
    return mfcc.T

#save mfcc to a json file
def save_mfcc(mfcc_data, mfcc_path):
    with open(mfcc_path, "w") as fp:
        json.dump(mfcc_data, fp, indent=4)

#extract mfcc feature from each file of the audio dataset and save to json files
def preprocess(dataset_path=DATASET_PATH):
    # mfcc_data = {}
    print(dataset_path)
    for path, subdirs, files in os.walk(dataset_path):
        dir_name = path.split('/')[-1]
        print(dir_name)
        json_dir = os.path.join(MFCC_PATH, dir_name)
        if not dir_name == 'audio':
            os.makedirs(json_dir, exist_ok=True)
        for f in files:
            audio_path = os.path.join(path, f)
            sample_key = f.split('.')[0]
            # print(path)
            # print(sample_key)
            mfcc = extract_mfcc(audio_path).tolist()
            mfcc_data = {'mfcc': mfcc}
            filename = os.path.join(json_dir, sample_key+'.json')
            save_mfcc(mfcc_data, filename)

if __name__ == "__main__":
    preprocess()
