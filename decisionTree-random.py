import numpy as np
import operator
import pandas as pd

def gini_score(groups, classes):
    n_samples = sum([len(group) for group in groups])
    gini = 0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        #print(size)
        for class_val in classes:
            #print(group.shape)
            p = (group[:,-1] == class_val).sum() / size
            score += p ** 2
        gini += (1.0 - score) * (size / n_samples)
        #print(gini)
    return gini


def split(feat, val, Xy):
    Xi_left = np.array([]).reshape(0, 80)
    Xi_right = np.array([]).reshape(0, 80)
    # subbcount = len(Xy)
    for i in Xy:
        if i[feat] <= val:
            Xi_left = np.vstack((Xi_left,i))
        if i[feat] > val:
            Xi_right = np.vstack((Xi_right,i))
    return Xi_left, Xi_right


def best_split(Xy, loop=None):
    classes = np.unique(Xy[:,-1])
    best_feat = 999
    best_val = 999
    best_score = 999
    best_groups = None
    count = Xy.shape[1]-1
    for feat in range(Xy.shape[1]-1):
        if loop == None and feat % 10 == 0:
            print('Split_branch: ' + str(count) + ' to go.')
        elif loop != None:
            print('Best_split: ' + str(count) + ' to go. In loop ' + str(loop+1) + '.')
        count -= 1
        # subcount = len(Xy)
        for i in Xy:
            # print(' ' + str(subcount))
            # subcount -= 1
            groups = split(feat, i[feat], Xy)
            #print(groups)
            gini = gini_score(groups, classes)
            #print('feat {}, valued < {}, scored {}'.format(feat,i[feat], gini))
            if gini < best_score:
                best_feat = feat
                best_val = i[feat]
                best_score = gini
                best_groups = groups
    output = {}
    output['feat'] = best_feat
    output['val'] = best_val
    output['groups'] = best_groups
    return output

def terminal_node(group):
    classes, counts = np.unique(group[:,-1], return_counts=True)
    return classes[np.argmax(counts)]

def split_branch(node, max_depth, min_num_sample, depth):
    left_node, right_node = node['groups']
    del(node['groups'])
    if not isinstance(left_node, np.ndarray) or not isinstance(right_node, np.ndarray):
        node['left'] = node['right'] = terminal_node(left_node + right_node)
        return
    if depth >= max_depth:
        node['left'] = terminal_node(left_node)
        node['right'] = terminal_node(right_node)
        return
    if len(left_node) <= min_num_sample:
        node['left'] = terminal_node(left_node)
    else:
        node['left'] = best_split(left_node)
        split_branch(node['left'], max_depth, min_num_sample, depth+1)
    if len(right_node) <= min_num_sample:
        node['right'] = terminal_node(right_node)
    else:
        node['right'] = best_split(right_node)
        split_branch(node['right'], max_depth, min_num_sample, depth+1)


def build_tree(Xy, max_depth, min_num_sample, i):
    root = best_split(Xy, loop=i)
    split_branch(root, max_depth, min_num_sample, 1)

    return root

def display_tree(node, depth=0):
    if isinstance(node,dict):
        print('{}[feat{} < {:.2f}]'.format(depth*'\t',(node['feat']+1), node['val']))
        display_tree(node['left'], depth+1)
        display_tree(node['right'], depth+1)
    else:
        # pass
        print('{}[{}]'.format(depth*'\t', node))


def predict_sample(node, sample):
    # print(node)
    if sample[node['feat']] < node['val']:
        if isinstance(node['left'],dict):
            return predict_sample(node['left'],sample)
        else:
            return node['left']
    else:
        if isinstance(node['right'],dict):
            return predict_sample(node['right'],sample)
        else:
            return node['right']


def predict(X):
    y_pred = np.array([])
    for i in X:
        y_pred = np.append(y_pred, predict_sample(tree,i))
    return y_pred.astype('int')

def accuracy(pred, true):
    correct = 0
    pred_len = len(pred)
    for i in range(pred_len):
        if pred[i] == true[i]:
            correct += 1
    return correct / pred_len


# Use Decision Tree built to train
data_test = np.array(pd.read_csv('testing.csv'))
data = np.array(pd.read_csv('training.csv'))

# The second column is English word, check if it is nan and change word to number
for i in range(len(data[:, 2])):
    if pd.isnull(data[i, 2]) == False:
        data[i, 2] = float(data[i, 2][1])
for i in range(len(data_test[:, 2])):
    if pd.isnull(data_test[i, 2]) == False:
        data_test[i, 2] = float(ord(data_test[i, 2][0])-65) * 10.0 + float(data_test[i, 2][1])

X_train_full = data[:, 1:80].astype('float')
for j in range(X_train_full.shape[1]):
    mean = np.nanmean(X_train_full[:, j], axis=0)
    for k in range(X_train_full.shape[0]):
        if pd.isnull(X_train_full[k][j]):
            X_train_full[k][j] = mean

y_test_array = 0
whole_array = 0
for i in range(19):
    indices = np.random.randint(0, data.shape[0], 200)
    X_train = X_train_full[indices, :].astype('float')
    X_test = data_test[:, 1:80].astype('float')

    a = pd.isnull(X_train)
    b = np.sum(a)
    print('b', b)
    y_train = data[indices, -1].astype('int')

    X = X_train
    y = y_train
    print('X_test', X, X.shape)
    print('y_test', y, y.shape)

    Xy = np.column_stack((X, y))
    tree = build_tree(Xy, 6, 50, i)
    y_pred = predict(X)
    # y_nons_pred = predict(data[401:, 3:80].astype('float'))
    # y_nons = data[401:, -1].astype('int')
    y_whole_pred_raw = predict(data[:, 1:80].astype('float'))
    y_whole_pred = np.reshape(y_whole_pred_raw, (1, y_whole_pred_raw.shape[0]))
    y_whole = data[:, -1].astype('int')
    y_test_pred = predict(X_test)
    y_test_pred = np.reshape(y_test_pred, (1, y_test_pred.shape[0]))
    # print('f', y_test_pred)
    if i == 0:
        y_test_array = y_test_pred
        whole_array = y_whole_pred
    else:
        y_test_array = np.concatenate((y_test_array, y_test_pred), axis=0)
        whole_array = np.concatenate((whole_array, y_whole_pred), axis=0)

    print('sample', i, accuracy(y_pred, y))
    # print('non-sample', accuracy(y_nons_pred, y_nons))
    print('whole', i, accuracy(y_whole_pred_raw, y_whole))

y_test_array = y_test_array.astype('int')
whole_array = whole_array.astype('int')
y_test_df = pd.DataFrame(y_test_array)
# y_test_array.to_csv('testing_loops.csv', index=False, header=False)
whole_df = pd.DataFrame(whole_array)
# whole_array.to_csv('training_loops.csv', index=False, header=False)
np.savetxt('y_result.csv', y_test_array.astype('int'), delimiter=',')
np.savetxt('whole.csv', whole_array.astype('int'), delimiter=',')
result = y_test_array.T
whole = whole_array.T
# print(result)
# print(y_test_array)

y_test_pred_mf = np.array([])
whole_pred_mf = np.array([])
for row in result:
    # print(row)
    counts = np.bincount(row)
    r = np.argmax(counts)
    y_test_pred_mf = np.append(y_test_pred_mf, r)

for row in whole:
    counts = np.bincount(row)
    r = np.argmax(counts)
    whole_pred_mf = np.append(whole_pred_mf, r)
print('whole_full', i, accuracy(whole_pred_mf, y_whole))

# Output prediction
y_pred_test = np.reshape(y_test_pred_mf, (y_test_pred_mf.shape[0],1))
y_ID = data_test[:,0]
y_ID = np.reshape(y_ID, (y_ID.shape[0],1))
y = np.append(y_ID, y_pred_test, axis=1).astype('int')

Title = np.array(['Id','Response'])
Title = np.reshape(Title, (1,2))
y = np.append(Title, y, axis=0)

my_df = pd.DataFrame(y)
my_df.to_csv('out.csv', index=False, header=False)
