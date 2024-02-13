from cgi import print_arguments
from statistics import mode
from traceback import print_tb
from matplotlib.pyplot import axis
import tensorflow.keras as keras
import tensorflow as tf
from function import *
from model import call_model
import argparse
from scipy.stats import pearsonr


'''
入力を受け取る
'''
parser = argparse.ArgumentParser(description='''
model training
''')
parser.add_argument("model", help="model name")
parser.add_argument("train_file", help="path of train file")
parser.add_argument("-l", '--send_line', help="send logs on line",
                    action='store_true')
parser.add_argument("-t", '--test', help="test mode",
                    action='store_true')
args = parser.parse_args()

model_name = args.model
input_file = args.train_file
l = args.send_line
test_mode = args.test
'''
定数の定義
'''

learning_rate=0.0001
ers_n = 5
epochs = 200
batch_size = 1280 
model = call_model(model_name)
now_date = making_now_date()
correct_TF = True
train_colum = '#m_expression_06.23.14.01'
weight_path = False
'''
データの読み込み
'''
print(f'{get_now_time()} プログラムを開始します')
send_line_notify('プログラムを開始します',l)



if not test_mode:
    df = pd.read_csv(input_file)
else:
    df = pd.read_csv(input_file)[:10000]


'''
調整領域
'''
df = df.sample(frac=1,random_state=0)
l_df = len(df)
s = len(df) * 8 //10
train_df = df[0:s]
val_df = df[s:]


x_train, t_train = input_data(train_df, data=train_colum)

x_val, t_val = input_data(val_df, data=train_colum)

if not test_mode:
    test_df = pd.read_csv('c_new_test_data.csv')
else:
    test_df = pd.read_csv('c_new_test_data.csv')[:10000]
    

x_test,t_test = input_data(test_df,data=train_colum)

x_val = [temp[40:440] for temp in x_val] 
x_test= [temp[40:440] for temp in x_test]

if 'xception' in model_name:
    model = model(weight = weight_path)
    x_val = change_for_xception(x_val)
    x_test = change_for_xception(x_test)



'''
モデルの構築
'''


# 早期終了のクラスを生成
ers = EarlyStoppingAndCorrecting(ers_n,correct_TF)
r_ers = EarlyStoppingAndCorrecting(ers_n,correct_TF)
mse = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)


def loss(t,y):
    return mse(t,y)
#def loss(t,y):
#    return tf.constant(- pearsonr(t, y)[0].astype(np.float32))


# 記録するオブジェクトの生成
train_loss = keras.metrics.Mean()
val_loss = keras.metrics.Mean()

# 訓練と検証における損失の推移を記録するオブジェクト
history = {'loss': [],  'val_loss': [],'train_R':[],'val_R':[] ,'test_R':[]}

'''
パラメーターの更新処理行う関数を定義
'''

def train_step(x, t):
    # 自動微分による勾配計算を記録するブロック
    with tf.GradientTape() as tape:
        # モデルに入力

        outputs = model(x)

        # 出力値と正解ラベルの誤差を計算
        tmp_loss = loss(t, outputs)

    # tapeに記録された操作をしようして誤差の勾配を計算
    grads = tape.gradient(
        # 現在のステップの誤差
        tmp_loss,
        # バイアス、重みのリストを取得
        model.trainable_variables
    )
    # 勾配降下法の更新式を適用して、バイアス・重みを更新
    optimizer.apply_gradients(
        (grad, var) for (grad, var) in zip(grads, model.trainable_variables) if grad is not None
    )

    # 精度と損失の記録
    train_loss(tmp_loss)
    return outputs

'''
検証データによりモデルを評価する関数の定義
'''

def val_step(x, t):
    # 検証データの予測値を取得
    preds = model(x)
    # 出力値と正解ラベルの誤差
    tmp_loss = loss(t, preds)
    # 損失と精度の記録
    val_loss(tmp_loss)
    return preds
'''
学習領域
'''

# 初期条件の設定

steps = len(x_train) // batch_size + 1
steps_val = len(x_val) // batch_size + 1

print(f'{get_now_time()} データの読み込みが完了しました。学習を開始します')
send_line_notify('学習を開始します',l)
p_time = pre_time(epochs)
correct_count = 0
axis_count = 0
axis_epochs = []
for epoch in range(epochs):
    train_recode = RecodePrediction()
    val_recode = RecodePrediction()
    x_,t_ = shuffle(x_train,t_train)
    x_ = slide2(x_)
    if 'xception' in model_name:
        x_ = change_for_xception(x_)

    # １ステップにおけるミニバッチ毎の学習
    for step in range(steps):
        start = batch_size * step
        end = start + batch_size
        # ミニバッチでバイアス・重みを更新
        train_recode(train_step(x_[start:end],t_[start:end]))
        
    # 検証データによるモデルの評価
    for step_val in range(steps_val):
        start = batch_size * step_val
        end = start + batch_size
        # 検証データのミニバッチ毎で損失と精度を測定
        val_recode(val_step(x_val[start:end], t_val[start:end]))
    
    
    # history に記録
    history['loss'].append(train_loss.result())
    history['val_loss'].append(val_loss.result())

    #p_ = get_result(model,x_)
    p_ = train_recode.get_prediction()
    rp = pearsonr(p_,t_)[0]
    
    #v_p_ = get_result(model,x_val)
    v_p_ = val_recode.get_prediction()
    v_rp = pearsonr(v_p_,t_val)[0]
    
    p_test = get_result(model,x_test)
    t_rp = pearsonr(p_test,t_test)[0]
    
    history['train_R'].append(rp)
    history['val_R'].append(v_rp)
    history['test_R'].append(t_rp)
    # 1エポックごとに結果を出力
    one_result = 'epoch({}[correct:{},axis:{}]) train_loss: {:.4}  val_loss: {:.4} train_R: {:.4} val_R: {:.4} test_R: {:.4}'.format(
              epoch+1,
              correct_count,
              axis_count,
              train_loss.result(),  # 訓練データの損失を出力
              val_loss.result(),   # 検証データの損失を出力
              rp,
              v_rp,
              t_rp
          )
    print(one_result)
    pt = p_time()
    
    # ラインに途中経過を送信します
    if (epoch + 1) % 1 == 0 or epoch == 0:
        line_result = '\nepoch({}) \ntrain_loss: {:.4}  \nval_loss: {:.4} \ntrain_R: {:.4} \nval_R: {:.4}\ntest_R: {:.4}\npredict_time:{}'.format(
            epoch+1,
            train_loss.result(),  # 訓練データの損失を出力
            val_loss.result(),   # 検証データの損失を出力
            rp,
            v_rp,
            t_rp,
            pt
        )
        send_line_notify(line_result,l)
    #correct,early_stop = ers(val_loss.result())
    correct,early_stop = ers(- v_rp)
    r_ers(- v_rp)
    # 一回目の学習の終了時にディレクトリの作成およびファイルの保存をします
    if epoch == 0:
        saving_dir = f'model/{now_date}'
        mkdir_if_none(saving_dir)
        # 学習に使った各種ファイルをコピー
        saving_files(saving_dir,now_date)
    if  correct:
        #axis_count += 1
        axis_epochs.append(epoch)
        #print(f'{get_now_time()} {axis_count}回目の軸の調整をします')
        #t_train = correct_axis(t_train, [p_, t_])
        #print(f'{get_now_time()} {axis_count}回目の軸の調整が終了しました')
        correct_count += 1
        print(f'{get_now_time()} {correct_count}回目のノイズの調整を行います')
        
        x_correct_train = [temp[40:440] for temp in x_train]
        if model_name == 'xception_1d':
            x_correct_train = change_for_xception(x_correct_train)
        x_correct_train = np.array(x_correct_train)
        p_correct_train = get_result(model,x_correct_train)
        p_correct_train,t_train = correcting_answer(p_correct_train,t_train,n = 10)
        #p_,t_ = correcting_answer(p_,t_)
        v_p_,t_val = correcting_answer(v_p_,t_val,n = 10)
        model.save_weights(
            f'{saving_dir}/weights_loss_{epoch}_{now_date}', save_format='tf')
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        print(f'{get_now_time()} {correct_count}回目のノイズの調整が終わりました')
    if early_stop:
        print('早期終了します')
        send_line_notify('早期終了します',l)
        line_result = '\nepoch({}) \ntrain_loss: {:.4}  \nval_loss: {:.4} \ntrain_R: {:.4} \nval_R: {:.4}\ntest_R: {:.4}\npredict_time:{}'.format(
            epoch+1,
            train_loss.result(),  # 訓練データの損失を出力
            val_loss.result(),   # 検証データの損失を出力
            rp,
            v_rp,
            t_rp,
            pt
        )
        send_line_notify(line_result, l)
        break
        
    # 成績が改善していた場合、重みを保存します
    if ers.min_or_not():
        
        model.save_weights(
            f'{saving_dir}/weights_loss_max_{now_date}', save_format='tf')
    if r_ers.min_or_not():
        
        model.save_weights(
            f'{saving_dir}/weights_{now_date}', save_format='tf')
    #else:
    #    axis_count += 1
    #    axis_epochs.append(epoch)
    #    print(f'{get_now_time()} {axis_count}回目の軸の調整をします')
    #    t_train = correct_axis(t_train,[p_,t_])
    #    print(f'{get_now_time()} {axis_count}回目の軸の調整が終了しました')


plot_history(history,saving_dir,now_date,axis_li = axis_epochs)

# それぞれの定数をメモとして残します
memo = f'{saving_dir}/memo_{now_date}.txt'
f = open(memo, 'w')
f.write(
    f'train_file={input_file}\n\
model={model_name}\n\
learning_rate={learning_rate}\n\
ers = {ers_n}\n\
epochs={epochs}\n\
result={one_result}\n\
correct={correct_TF}\n\
train_colum={train_colum}'
)
f.close()

# 散布図を作成
plot_saving_path = f'{saving_dir}/train_val_plot_{now_date}.png'
train_val_map([p_,t_],[v_p_,t_val], plot_saving_path,r_li=[rp,v_rp])
train_plot_path = f'{saving_dir}/train_plot_{now_date}.png'
plot_map(p_,t_,train_plot_path,r=rp)

val_plot_path = f'{saving_dir}/val_plot_{now_date}.png'
plot_map(v_p_,t_val,val_plot_path,tvt='val',r=v_rp)

test_plot_path = f'{saving_dir}/test_plot_{now_date}.png'
plot_map(p_test, t_test, test_plot_path, tvt='test', r=t_rp)

# ここまで到達したら、finish_modelにコピー
original_dir = f'model/{now_date}'
copy_dir = f'finish_model/{now_date}'
subprocess.run(f'cp -r {original_dir} {copy_dir}', shell=True)

print(f'{get_now_time()} プログラムを終了します')
