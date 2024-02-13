from function import *

'''
入力を受け取る
'''
parser = argparse.ArgumentParser(description='''
model training
''')
parser.add_argument("model_name", help="model name")
parser.add_argument("weight_path", help="weight_path")
args = parser.parse_args()

model_name = args.model_name
weight_path = args.weight_path



df = pd.read_csv('test_sequences.txt',
                 delimiter='\t',
                 names=['#sequence', '#expression'])


reproduce_modle = GetReproduceModel(model_name,weight_path)
model = reproduce_modle()

def test_one_hot_list(df):
    front = 'TTTTCACATC'
    back =  'GGTTACGGCTGTTTCTTAAT'
    seq_li = df['#sequence'].to_list()
    seq_80_li = [temp[17:-13] for temp in seq_li]
    seq_120_li = [front + temp + back for temp in seq_80_li]
    dic = {'A' : '1000',
           'C' : '0100',
           'G' : '0010',
           'T' : '0001',
           'N' : '0000',}
    
    all_seq = '/'.join(seq_120_li)
    
    for key,value in dic.items():
        all_seq = all_seq.replace(key,value)
    
    one_hot_li = all_seq.split('/')
    x_li = [temp + '0' * (400 - len(temp)) for temp in one_hot_li]
    x_li = [temp[:400] for temp in x_li]
    x_li = [list(temp) for temp in x_li]
    x_li = np.array(x_li, dtype='float32')
    return x_li

x_li = test_one_hot_list(df)
x_li = change_for_xception(x_li)
p_li = get_result(model,x_li)
p_li = list(np.array(p_li))

submission_li = [f'"{int(i)}": {p_li[i]}' for i in range(len(p_li))]


s = ', '.join(submission_li)
s = '{' + s + '}'

# weight_path = finish_model/07.13.01.49/weights_loss_102_0_07.13.01.49.data-00000-of-00001
weight_name = weight_path.split('/')[-1]
model_dir = weight_path.replace(f'/{weight_name}','')
save_path = f'{model_dir}/submission_{weight_name}.json'

f = open(save_path, 'w')

f.write(s)

f.close()
