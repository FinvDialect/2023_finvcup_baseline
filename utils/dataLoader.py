'''
DataLoader for training
'''
import pandas as pd
import glob, numpy, os, random, soundfile, torch
from scipy import signal

class train_loader(object):
    def __init__(self, train_list, num_frames, **kwargs):
        self.num_frames = num_frames
        # Load data & labels
        self.df_train = pd.read_csv(train_list)
        print("train_data_shape:", self.df_train.shape)
        self.data_list = self.df_train['wav_path'].tolist()
        self.data_label = self.df_train['label'].tolist()

    def __getitem__(self, index):
        # Read the utterance and randomly select the segment
        audio, sr = soundfile.read(self.data_list[index])
        length = self.num_frames * 80 
        #data augment
        while len(audio)<=length:
            df_tmp = self.df_train[self.df_train["label"]==self.data_label[index]]
            df_tmp = df_tmp.sample(n=1)
            audio_2, sr2 = soundfile.read(df_tmp["wav_path"].tolist()[0])
            audio = numpy.concatenate((audio,audio_2))

        
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio],axis=0)
        return torch.FloatTensor(audio[0]), self.data_label[index]

    def __len__(self):
        return len(self.data_list)
    
class eval_loader(object):
    def __init__(self, eval_list, **kwargs):
        # Load data & labels
        df_eval = pd.read_csv(eval_list)
        print("valid_data_shape:", df_eval.shape)
        self.data_list = df_eval['wav_path'].tolist()
        self.data_label = df_eval['label'].tolist()

    def __getitem__(self, index):
        # Read the utterance
        audio, sr = soundfile.read(self.data_list[index])
        audio = numpy.stack([audio],axis=0)
        return torch.FloatTensor(audio[0]), self.data_label[index]

    def __len__(self):
        return len(self.data_list)
    
def my_collate_fn(data, max_length=12000):

    lens = [x[0].shape[0] for x in data]
    max_len = max(lens)
    print(max_len)
    max_length = min(max_len, max_length)

    features = torch.zeros(len(data), max_length)
    for i, length in enumerate(lens):
        features[i,:length] = data[i][0][:max_length]    
    labels = torch.tensor([x[1] for x in data])

    return features, labels

if __name__ == '__main__':
    eval_loader = eval_loader('./data_0609/test2','.')
    evalLoader = torch.utils.data.DataLoader(eval_loader, batch_size = 256, shuffle = True, num_workers = 4, collate_fn=my_collate_fn)
    for data, label in evalLoader:
        print(data.shape)
        print(label.shape)
