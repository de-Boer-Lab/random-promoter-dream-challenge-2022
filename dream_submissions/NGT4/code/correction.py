from function import *

file_path = ['c_new_mini_train_data.csv',
             'c_new_test_data.csv', 'c_new_train_data.csv']

for path in file_path:
    df = pd.read_csv(path)
    reproduce_model = GetReproduceModel('xception_1d', 'finish_model/06.23.14.01')
    p_li, t_li = reproduce_model.get_predict_expression(
        path,  '#m_correction_expression')
    df = pd.read_csv(path)

    df['#m_expression_06.23.14.01'] = list(p_li.numpy())
    df_0 = df[df['#just'] == 0]
    df_1 = df[df['#just'] == 1]


    df_0 = df_0.sort_values(
        by=['#expression', '#m_expression_06.23.14.01']).reset_index(drop=True)

    li_p = []

    for i in range(18):

        df_m = df_0[df_0['#expression'] == i].reset_index(drop=True)
        df_m['#m_expression_06.23.14.01'] = df_m.index
        l = len(df_m) - 1
        df_m['#m_expression_06.23.14.01'] = df_m['#m_expression_06.23.14.01'] / l + i - 0.5
        li_m = df_m['#m_expression_06.23.14.01'].to_list()
        li_p += li_m


    df_0['#m_expression_06.23.14.01'] = li_p

    df_1['#m_expression_06.23.14.01'] = df_1['#expression']
    df = pd.concat([df_0, df_1])
    df = df.sort_values(
        by=['#m_expression_06.23.14.01']).reset_index(drop=True)
    df['#m_expression_06.23.14.01'] = df.index
    df['#m_expression_06.23.14.01'] /= len(df)

    df.to_csv(path, index=False)
    print(f'{path}が終了')
send_line_notify('プログラムが終了')
