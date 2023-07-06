import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import torch
from utils.ECAPAModel import ECAPAModel, ECAPA_TDNN
import argparse
import os
SEED = 42
def load_model(model_path, C=1024):
    model = ECAPA_TDNN(C=C)

    self_state = model.state_dict()
    loaded_state = torch.load(model_path)
    for name, param in loaded_state.items():
        origname = name
        if name not in self_state:
            name = name.replace("dialect_encoder.", "")
            if name not in self_state:

                continue
        if self_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
            continue
        self_state[name].copy_(param)

    model.load_state_dict(self_state)
    model.eval()
    return model
                  

def compute_pair_distance(model, pair_df, device='cuda',res_path='submit.csv'):
    model.to(device)
    id2path = {}
    ids = set(pair_df.id1.tolist()+pair_df.id2.tolist())
    for r,_,files in os.walk('data'):
        for fname in files:
            if fname.endswith('.wav'):
                fid = fname.split('.')[0]
                fpath = os.path.join(r,fname)
                id2path[fid] = fpath
    pair_df['wav_path1'] = pair_df.id1.apply(lambda x:id2path[x])
    pair_df['wav_path2'] = pair_df.id2.apply(lambda x:id2path[x])
    
    embeddings = {}
    res = []
    for wav_id in tqdm(ids):
        wav_path = id2path[wav_id]
        audio, sr = sf.read(wav_path)
        audio = np.stack([audio],axis=0)
        audio = torch.FloatTensor(audio[0]).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.forward(audio, aug=False)
        embeddings[wav_id] = embedding

    for i, row in tqdm(pair_df.iterrows()):
        dist = torch.cdist(embeddings[row['id1']], embeddings[row['id2']], p=2)
        res.append(dist.item())
    pair_df['distance']=res
    pair_df[['distance']].to_csv(res_path, encoding='utf8',index=False)
    return pair_df

def main():
    parser = argparse.ArgumentParser(description = "ECAPA_trainer")
    parser.add_argument('--device',      type=str,   default='cuda:0',       help='Device model inferring on ')
    parser.add_argument('--model_path',      type=str,   default='exps/model/model_0002.model',       help='Model checkpoint path')
    parser.add_argument('--test_path',      type=str,   default='data/test_pair',       help='Path of test file, strictly same with the original file')
    parser.add_argument('--save_path', type=str, default='submit.csv', help='Path of result')
    args = parser.parse_args()
    print('loading model...')
    model = load_model(args.model_path)
    pair_df = pd.read_csv(args.test_path)
   
    print('model inferring...')
    compute_pair_distance(model, pair_df, device=args.device, res_path=args.save_path)


if __name__ == '__main__':
    main()
