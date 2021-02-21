import pandas as pd
import numpy as np
import scipy.sparse as sp

def data_preprocess(path, train_ratio = 0.8, binary_threshold = 0.0):
    rating_df = pd.read_csv(path, delimiter="::", 
        names = ['user_id', 'movie_id', 'ratings', 'timestamp'],
        dtype= {
            'user_id': int,
            'movie_id': int,
            'ratings': float,
            'timestamp': float
        },
        engine = 'python'
    )
    num_users = len(pd.unique(rating_df.user_id))
    num_items = len(pd.unique(rating_df.movie_id))
    print (f"[*] # Users : {num_users}, # Items: {num_items}")
    if binary_threshold > 0.0:
        print (f"[*] Binary_threshold is {binary_threshold}. ratings >= {binary_threshold} will be adopted.")
        rating_df = rating_df[rating_df.ratings >= binary_threshold]
    rating_df['ratings'] = 1.0

    print ('[*] Assign new user id..')

    num_items_per_user = rating_df.groupby('user_id', as_index=False).size().set_index('user_id')

    num_items_per_user.columns = ['item_cnt']
    num_items_per_user['new_id'] = list(range(num_users))
    user_frame = num_items_per_user

    user_dict = user_frame.to_dict()
    user_id_dict = user_dict['new_id']
    user_frame = user_frame.set_index('new_id')
    user_to_num_items = user_frame.to_dict()['item_cnt']

    rating_df.user_id = [user_id_dict[x] for x in rating_df.user_id.tolist()]

    print ('[*] Assign new movie id..')
    num_users_per_item = rating_df.groupby('movie_id', as_index=False).size().set_index('movie_id')
    num_users_per_item.columns = ['user_cnt']
    num_users_per_item['new_id'] = list(range(num_items))
    item_frame = num_users_per_item

    frame_dict = item_frame.to_dict()
    item_id_dict = frame_dict['new_id']
    item_frame = item_frame.set_index('new_id')
    item_to_num_users = item_frame.to_dict()['user_cnt']

    rating_df.movie_id = [item_id_dict[x] for x in rating_df.movie_id.tolist()]

    print ('[*] Split data into train / test')

    rating_group = rating_df.groupby('user_id')

    train_list, test_list = [], []
    for _, group in rating_group:
        user = pd.unique(group.user_id)[0]
        num_items_user = len(group)
        num_train = int(train_ratio * num_items_user)
        num_test = num_items_user - num_train

        group = group.sort_values(by='timestamp')
        num_zero_train, num_zero_test = 0, 0
        idx = np.ones(num_items_user, dtype='bool')

        test_idx = np.random.choice(num_items_user, num_test, replace=False)
        idx[test_idx] = False
        
        if len(group[idx]) == 0:
            num_zero_train += 1
        else:
            train_list.append(group[idx])

        if len(group[np.logical_not(idx)]) == 0:
            num_zero_test += 1
        else:
            test_list.append(group[np.logical_not(idx)])

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    train_sparse = df_to_sparse(train_df, shape=(num_users, num_items))
    test_sparse = df_to_sparse(test_df, shape=(num_users, num_items))

    return train_sparse, test_sparse


def df_to_sparse(df, shape):
    rows, cols = df.user_id, df.movie_id
    values = df.ratings
    inter_matrix = sp.csr_matrix((values, (rows, cols)), dtype='float64', shape=shape)
    num_nonzeros = np.diff(inter_matrix.indptr)
    rows_to_drop = num_nonzeros == 0
    if sum(rows_to_drop) > 0:
        print(f'{ sum(rows_to_drop)} empty users are dropped from matrix.')
        inter_matrix = inter_matrix[num_nonzeros != 0]
    return inter_matrix