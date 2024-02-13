from function import *

front = 'AACTGCATTTTTTTCACATC'
back =  'GGTTACGGCTGTTTCTTAAT'

# ファイルの読み込み
df = pd.read_csv('train_sequences.tsv',
                 delimiter='\t',
                 names=['#sequence', '#expression'])



# one_hot と len
seq_li = df['#sequence'].to_list()
seq_80_li = [temp[17:-13] for temp in seq_li]
seq_130_li = [front + temp + back for temp in seq_80_li]
one_hot_df = get_one_hot(seq_130_li)
print('one-hot化が終了')
df = pd.concat([df, one_hot_df], axis=1)

# just
expression_li = df['#expression'].to_list()
just_li = [get_just(temp) for temp in expression_li]
print('justが終了')
just_df = pd.DataFrame(just_li, columns=['#just'])
df = pd.concat([df, just_df], axis=1)
seq80_df = pd.DataFrame(seq_80_li, columns=['#sequence80'])
df = pd.concat([df, seq80_df], axis=1)
seq120_df = pd.DataFrame(seq_130_li, columns=['#sequence120'])
df = pd.concat([df, seq120_df], axis=1)

# 並び替え
df = df[['#sequence', '#sequence80', '#sequence120', '#one_hot_sequence', '#expression', '#len', '#just']]

df.to_csv('new_all_data.csv', index=False)

print(df)

