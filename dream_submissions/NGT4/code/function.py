from ast import Str
from cProfile import label
import re
import collections
import pandas as pd
import time
import datetime
import math
import requests
from sympy import N
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from model import call_model
import matplotlib.pyplot as plt
import os
import subprocess
import argparse
from scipy.stats import pearsonr

def send_line_notify(notification_message, l=True):
    """
    LINEに通知する
    """
    try:
        if l:
            line_notify_token = '6QgQruJbq34EnUjuc4wk06lgL4ifx4MsWNyNbs4C4xw'
            line_notify_api = 'https://notify-api.line.me/api/notify'
            headers = {'Authorization': f'Bearer {line_notify_token}'}
            data = {'message': f'message: {notification_message}'}
            requests.post(line_notify_api, headers=headers, data=data)
    except Exception as e:
        print(e)
        pass

class pre_time:
    def __init__(self, all):
        self.start = time.time()
        self.count = self.start
        self.memo = 'start'
        self.n = 0
        self.all = all

    def __call__(self):
        self.n += 1
        now_time = time.time()
        one_epoch = (now_time - self.start) / self.n
        predict_time = self.start + one_epoch * self.all 
        predict_time = datetime.datetime.fromtimestamp(predict_time)
        predict_time = predict_time.strftime('%m/%d %H:%M')
        now_time = datetime.datetime.fromtimestamp(now_time)
        now_time = now_time.strftime('%m/%d %H:%M')
        print(f'{now_time} {self.n}/{self.all}({self.n / self.all * 100}%)の処理が終了')
        print(f'予想終了時刻 {predict_time} ({one_epoch/60 :.3}分/epoch)')
        return predict_time

    def time_count(self, memo):
        count_now_time = time.time()
        t = count_now_time - self.count
        print(f'[{self.memo}-{memo}]: {t}')
        self.count = count_now_time
        self.memo = memo

def get_now_time():
    now_time = time.time()
    now_time = datetime.datetime.fromtimestamp(now_time)
    now_time = now_time.strftime('%m/%d %H:%M')
    return now_time

def get_one_hot(seq_li,N_n = 40 ):
    '''
    入力: list [n]
    出力: pd.dataframe 
    
    A -> [1,0,0,0],C -> [0,1,0,0],G -> [0,0,1,0],T -> [0,0,0,1],N -> [0,0,0,0]
    に変換します（参考　https://www.ddbj.nig.ac.jp/activities/training/2016-06-27.html）
    
    '''
    
    dic = {'A' : '1000',
           'C' : '0100',
           'G' : '0010',
           'T' : '0001',
           'N' : '0000',}
    
    all_seq = '/'.join(seq_li)
    
    for key,value in dic.items():
        all_seq = all_seq.replace(key,value)
    
    one_hot_li = all_seq.split('/')
    one_hot_li = [temp + '0' * (N_n * 4) for temp in one_hot_li]
    one_hot_li = [[temp,len(temp) / 4 - (N_n + 40)] for temp in one_hot_li]
    one_hot_df = pd.DataFrame(one_hot_li,columns=['#one_hot_sequence','#len'])
    return one_hot_df

def get_just(x):
    '''
    .000000 なら０
            でないなら1
    '''
    a = math.floor(x)
    if x == a:
        return 0
    else:
        return 1

'''
training
'''

# expressionが均等になるように学習させる
def get_same_percentage(df: pd.DataFrame, n=10000):
    df = df.sample(frac=1)
    for i in range(17):
        df_m = df[(df['#expression'] > i) &
                  (df['#expression'] < (i + 1))]
        df_m = df_m[0:n]
        if not i:
            data = df_m
        else:
            data = data.append(df_m)
    data = data.sample(frac=1)
    data = data.reset_index(drop=True)
    return data

def input_data(df,data,a = 120 * 4, n = 1):

    x_li = df['#one_hot_sequence'].to_list()
    x_li = [list(temp[:a]) for temp in x_li]
    x_li = np.array(x_li, dtype='float32')
    t_li = df[data].to_list()
    t_li = np.array(t_li, dtype='float32') / n
    return x_li ,t_li


def adjustment_data(train_df):
    df_li = []
    for i in range(17):
        if i != 16:
            df_m = train_df[(train_df['#expression'] >= i) &
                            (train_df['#expression'] < (i+1))]
        else:
            df_m = train_df[(train_df['#expression'] >= i) &
                            (train_df['#expression'] <= (i+1))]
        df_li.append(df_m)

    df_len = [len(temp) for temp in df_li]
    len_mean = sum(df_len) / len(df_len)

    # 平均に満たない場合は、平均に到達するように設定する
    adjust_li = [(len_mean // temp) + 1 for temp in df_len]
    # ただし、10倍を超えるものは１０倍で統一する
    limit = 3

    def adf(x):
        if x > limit:
            return limit
        elif x == 0:
            return 1
        else:
            return x
    adjust_li = [adf(temp) for temp in adjust_li]

    final_df_li = []
    for i in range(17):
        n = adjust_li[i]
        df_m = df_li[i]
        final_df_li += [df_m] * int(n)

    train_df = pd.concat(final_df_li)
    return train_df

# 配列を左右にスライドさせて過学習を防ぐ
def slide(x_li, n=5):
    def one_slide(x , i):
        if i > 0:
            # 左側を4i個削除
            x = x[i * 4:]
            # 右側に4i個の０を追加
            new_li = np.full((i * 4), 0)
            x = np.concatenate([x, new_li], 0)
        elif i < 0:
            # 右側を4i個削除
            x = x[:i * 4]
            # 左側に4iの０を追加
            new_li = np.full((i * -4), 0)
            x = np.concatenate([new_li, x], 0)
        return x
    min_slide = n * -1
    max_slide = n + 1
    r = np.random.randint(min_slide, max_slide, (len(x_li)))
    x_li = [one_slide(x_li[n], r[n]) for n in range(len(x_li))]
    
    return np.array(x_li, dtype='float32')


def slide2(x_li, n=5):
    def one_slide(x, i):
        x = x[40 + i * 4 : 440 + i * 4]
        return x
    min_slide = n * -1
    max_slide = n + 1

    r = np.random.randint(min_slide, max_slide, (len(x_li)))
    x_li = [one_slide(x_li[n], r[n]) for n in range(len(x_li))]
    return np.array(x_li, dtype='float32')

# 早期学習終了
class EarlyStoppingAndCorrecting:
    def __init__(self, patience,correct_TF):
        self.epoch = 0
        self.correct_count = 0
        self.pre_loss = float('inf')
        self.patience = patience
        self.TF = True
        self.correct_TF = correct_TF
        self.mini = float('inf')
    
    def __call__(self,current_loss):
        self.current_loss = current_loss
        if self.mini >= self.current_loss:
            self.mini = self.current_loss
            self.TF = True
        else:
            self.TF = False
        if self.correct_TF:
            return self.esag()
        else:
            return self.es()
    def es(self):
        if self.pre_loss < self.current_loss:
            self.epoch += 1
            if self.epoch == self.patience:
                self.epoch = 0
                return False,True
        else:
            self.epoch = 0
            self.pre_loss = self.current_loss
        return False,False
    
    def esag(self):
        if self.pre_loss < self.current_loss:
            self.epoch += 1
            if self.epoch == self.patience:
                self.epoch = 0
                # 3回目以降の修正は行わない
                self.correct_count += 1
                if self.correct_count == 3:
                    # 修正：False,早期終了：True
                    return False, True
                # 修正：True,早期終了：False
                return True,False
        else:
            self.epoch = 0
            self.pre_loss = self.current_loss
        return False,False

    def min_or_not(self):
        return self.TF

def plot_history(history, saving_dir,now_date,axis_li = []):
    axis_point_x = []
    axis_point_y = []
    for i in axis_li:
        for temp in ['train_R','val_R']:
            axis_point_x.append(i - 1)
            axis_point_y.append(history[temp][i - 1])
            
    plt.plot(history['train_R'], "-", label="train_R")
    plt.plot(history['val_R'], "-", label="val_R")
    plt.plot(history['test_R'], "-", label="test_R")
    plt.scatter(axis_point_x, axis_point_y, c="k", label='axis_correct')
    plt.title(f'R_{now_date}')
    plt.xlabel('epoch')
    plt.ylabel('R')
    plt.legend(loc="lower right")
    plt.savefig(f'{saving_dir}/R_{now_date}.png')
    plt.clf()
    # 損失の履歴をプロット
    axis_point_x = []
    axis_point_y = []
    for i in axis_li:
        for temp in ['loss', 'val_loss']:
            axis_point_x.append(i - 1)
            axis_point_y.append(history[temp][i - 1])
    plt.plot(history['loss'], "-", label="train_loss",)
    plt.plot(history['val_loss'], "-", label="val_loss")
    plt.scatter(axis_point_x, axis_point_y, c="k",label='axis_correct')
    plt.title(f'loss_{now_date}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig(f'{saving_dir}/loss_{now_date}.png')
    
# 日時を取得
def making_now_date():
    dt_now = datetime.datetime.now()
    return dt_now.strftime('%m.%d.%H.%M')

# dir がなかったら作成
def mkdir_if_none(path):
    if not os.path.exists(path):
        os.mkdir(path)

# 検証のために、モデルの学習のたびに主要ファイルをコピーする
def saving_files(saving_dir,now_date):
    code_li =  ['training', 'function', 'model']
    for file in code_li:
        original_file = f'code/{file}.py'
        copy_path = f'{saving_dir}/{file}_{now_date}.py'
        
        subprocess.run(f'cp {original_file} {copy_path}', shell=True)

# 予測値を取得
def get_result(model,x,batch_size = 1280):
    p_li = []
    l_x = len(x)
    steps = l_x // batch_size + 1
    for step in range(steps):
        start = batch_size * step
        end = start + batch_size
        p_m = model(x[start:end])
        
        if not step:
            p_li = p_m 
        else:
            p_li = tf.concat([p_li, p_m], 0)
    
    return p_li

# 散布図の作成
def plot_map(p_li,t_li,path,tvt='train',title = '',r = ''):
    a, b = np.polyfit(p_li, t_li, 1)

    y = a * p_li + b
    color_dict = {'train':['c','b'],
                  'val':['orange','r'],
                  'test':['lime','g']}
    color = color_dict[tvt]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(p_li, t_li, s=0.03, c=color[0])
    ax.plot(p_li,p_li,c = "k")
    ax.plot(p_li,y,c = color[1])
    if not title:
        title = path.split('/')[-1]
    plt.title(title)
    plt.xlabel('predicted expression level')
    plt.ylabel('expression level')
    plt.text(0.8,0,f'R = {r:.4}')
    plt.savefig(path)


def train_val_map(train_li, val_li, saving_path, r_li=['', ''],title=''):
    train_p ,train_t = train_li
    val_p ,val_t = val_li
    train_r ,val_r = r_li
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lis = [[train_p,train_t,'c','b'],
        [val_p,val_t,'orange','r']]
    for li in lis:
        p_li,t_li,c = li[0:3]
        ax.scatter(p_li, t_li, s=0.03,c=c)
    y = train_p
    ax.plot(train_p, y, c='k')
    if not title:
        title = saving_path.split('/')[-1]
    plt.title(title)
    plt.xlabel('predicted expression level')
    plt.ylabel('expression level')
    if r_li[0]:
        plt.text(0.8, 0, f'train_R = {train_r:.4}\nval_R={val_r: .4}',linespacing = 1.3)
    plt.savefig(saving_path)

class GetReproduceModel:
    def __init__(self,model_name,weight_path):
        self.model_name = model_name
        model = call_model(model_name)
        if 'xception' in self.model_name:
            model = model()
        model.compile(optimizer='adam')
        self.weights_path = weight_path
        #weight_date = weight_path.split('/')[-1]
        #self.weights_path = f'{weight_path}/weights_{weight_date}'
        model.load_weights(self.weights_path)
        model.build(input_shape=(None, 100,4,1))
        self.model = model
    def __call__(self):
        return self.model
    def get_predict_expression(self,data_path,colum,test= False):
        if not test:
            df = pd.read_csv(data_path)
        else:
            df = pd.read_csv(data_path)[:1000]
        x_li, t_li = input_data(df, colum,n =1)
        x_li = [temp[40:440] for temp in x_li]
        if 'xception' in self.model_name:
            x_li = change_for_xception(x_li)
        p_li = get_result(self.model,x_li)
        return p_li,t_li


def correcting_answer(p_li, t_li, n=20):
    def correcting_one_answer(p, t, n):
        t = t - (t - p) / n
        return t
    t_li = [correcting_one_answer(p_li[i], t_li[i], n=n)
            for i in range(len(t_li))]

    return p_li, t_li
def change_for_xception(x_):
    x_ = np.array(x_)
    x_ = x_.reshape([-1,100,4])
    return x_

class RecodePrediction:
    def __init__(self):
        self.first = True
    def __call__(self,p_):
        if self.first:
            self.pre_li = p_
            self.first = False
        else:
            self.pre_li = tf.concat([self.pre_li,p_],0)
    def get_prediction(self):
        return self.pre_li



def correct_axis(t_li, li):
    t_li = np.array(t_li)
    p_,t_ = li
    a, b = np.polyfit(p_, t_, 1)
    t_li = (t_li - b) / a
    return t_li


def one_search(x: str,word,Len = 100):
    X_len = len(x)
    i = 0
    n = len(word) - 1
    li = []
    while i != -1:
        i = x.rfind(word)
        x = x[:i + n ]
        li.append(i)
    li = li[1:]
    one_hot_li = [0 for i in range(X_len)]
    for i in li:
        one_hot_li[i] = 1
    
    one_hot_li2 = [one_hot_li[i * Len:(i+ 1) * Len]
                   for i in range(int(X_len / Len))]
    return one_hot_li2


def search(li,Len = 80):
    li = [temp + 'N' * (Len - len(temp)) for temp in li]
    x = ''.join(li)
    li_4 = ['A', 'T', 'C', 'G']
    li_256 = ['A', 'T', 'C', 'G']
    for i in range(3):
        for i in range(len(li_256)):
            t = li_256[i]
            li_256[i] = [t + temp for temp in li_4]
        li_256 = list(np.array(li_256).reshape(-1))
    finish_li = []
    for temp in li_256:
        finish_li.append(one_search(x, temp,Len=Len))
    finish_li2 = []
    for i in range(len(li)):
        finish_li2.append([temp[i] for temp in finish_li])
    return finish_li2



