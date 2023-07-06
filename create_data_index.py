import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

SEED = 42
def main():
    label_dic = {
            '普通话': 0,
            '成都': 1,
            '郑州': 2,
            '武汉': 3,
            '广州': 4,
            '上海': 5,
            '杭州': 6,
            '厦门': 7,
            '长沙': 8,
    }
    ids = []
    paths = []
    labels = []
    for r, _, files in tqdm(os.walk('./data/train')):
        for fname in files:
            if fname.endswith('.wav'):
                label = label_dic[r.split('/')[-1]]
                fid = fname.split('.')[0]
                fpath = os.path.join(r, fname)

                ids.append(fid)
                paths.append(fpath)
                labels.append(label)

    train_df = pd.DataFrame({
        'id': ids,
        'wav_path': paths,
        'label': labels
    })
    train_df = train_df.sample(frac=1,random_state=SEED)
    new_train_df, valid_df = train_test_split(train_df, test_size=0.2,random_state=SEED)
    new_train_df.to_csv('data/train_df', index=False, encoding='utf8')
    valid_df.to_csv('data/valid_df', index=False, encoding='utf8')

if __name__ =='__main__':
    main()