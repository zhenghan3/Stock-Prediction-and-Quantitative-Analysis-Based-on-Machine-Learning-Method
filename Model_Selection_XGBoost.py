# Model selection process of the date in the date_list

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
import time
from matplotlib.pyplot import MultipleLocator

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate, PredefinedSplit

from Data_Cleaning import save_to_file, read_from_file, get_dataset_bydate, get_date_list, authorization_jq, get_dataset_bydate_dev

para_dict = {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}
# model.kwargs['tree_method']='auto'

# Final model of 1 year
xgb_model1 = XGBClassifier(
    learning_rate=0.04,
    n_estimators=200,
    max_depth=8,
    min_child_weight=4,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.005,
    reg_lamda=1,

    objective='binary:logistic',
    nthread=-1,
    scale_pos_weight=1,
    seed=27,
    tree_method=para_dict['tree_method'],
    predictor=para_dict['predictor']
)

# Final model of 2 years
xgb_model2 = XGBClassifier(
    learning_rate=0.24,
    n_estimators=200,
    max_depth=8,
    min_child_weight=2,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1e-6,
    reg_lamda=1,

    objective='binary:logistic',
    nthread=-1,
    scale_pos_weight=1,
    seed=27,
    tree_method=para_dict['tree_method'],
    predictor=para_dict['predictor']
)

# Final model of 3 years
xgb_model3 = XGBClassifier(
    learning_rate=0.1,
    n_estimators=200,
    max_depth=7,
    min_child_weight=4,
    gamma=0,
    subsample=0.75,
    colsample_bytree=0.85,
    reg_alpha=1e-5,
    reg_lamda=0.8,

    objective='binary:logistic',
    nthread=-1,
    scale_pos_weight=1,
    seed=27,
    tree_method=para_dict['tree_method'],
    predictor=para_dict['predictor']
)

xgb_model = [xgb_model1, xgb_model2, xgb_model3]


'''
# Save the file to google drive
def save_to_google_drive(interval=6, force_save=False):
    global sum
    sum += 1
    if (sum % interval == 0) or (force_save):
        if not os.path.exists("/content/drive/My Drive/Colab Notebooks/Dissertation/Factor/Database/dataset"):
            os.makedirs("/content/drive/My Drive/Colab Notebooks/Dissertation/Factor/Database/dataset")

        if not os.path.exists("/content/drive/My Drive/Colab Notebooks/Dissertation/Factor/Database/result"):
            os.makedirs("/content/drive/My Drive/Colab Notebooks/Dissertation/Factor/Database/result")


        # From Colab to Google drive
        !cp - rfn
        '/content/Database/result' "/content/drive/My Drive/Colab Notebooks/Dissertation/Factor/Database"
        !cp - rfn
        '/content/Database/dataset' "/content/drive/My Drive/Colab Notebooks/Dissertation/Factor/Database"
'''

# Function to print the time
def print_time(start_time, s='', wrap_above=True):
    if wrap_above:
        print('\nElapsed ' + s + 'time: %.2f minutes\n' % ((time.time() - start_time) / 60.0))
    else:
        print('Elapsed ' + s + 'time: %.2f minutes\n' % ((time.time() - start_time) / 60.0))


# Set the gpu acceleration status of the model
def model_ungpu(model, status=True):
    if not status:
        model.kwargs['tree_method'] = para_dict['tree_method']
        model.kwargs['predictor'] = para_dict['predictor']

    else:
        model.kwargs['tree_method'] = 'auto'
        if 'predictor' in model.kwargs:
            del (model.kwargs['predictor'])
    return model


# Function to slice the list within the range
def list_slice(li, start, end):
    start_index = li.index(start)
    end_index = li.index(end) + 1
    return li[start_index:end_index]


# Function to obtain the training, test and dev set with the set dev_proportion
def get_datasetfull_bydate(current_date, date_list, interval=20, full=False):
    global drop
    training_set1, dev_set_index1, test_set = get_dataset_bydate(current_date, date_list, interval=interval, dev_set=True,
                                                                 year=1, drop=drop)
    if full:

        training_set2, dev_set_index2, test_set2 = get_dataset_bydate(current_date, date_list, interval=interval, dev_set=True,
                                                                      year=2, drop=drop)
        training_set3, dev_set_index3, test_set3 = get_dataset_bydate(current_date, date_list, interval=interval, dev_set=True,
                                                                      year=3, drop=drop)
        return [training_set1, training_set2, training_set3], [dev_set_index1, dev_set_index2, dev_set_index3], test_set
    else:
        return [training_set1, training_set1, training_set1], [dev_set_index1, dev_set_index1, dev_set_index1], test_set


# Function to split the dataset into X and Y
def xy_split(training_set):
    X_train = training_set.drop(['pchg', 'label'], axis=1)
    Y_train = training_set['label']
    return X_train, Y_train


# Function to split the training, test, dev set into X and Y
def xy_split_all():
    global test_set, training_set_full
    X_test, Y_test = xy_split(test_set)
    X_train_full, Y_train_full = [None] * 3, [None] * 3
    X_train_full[0], Y_train_full[0] = xy_split(training_set_full[0])
    X_train_full[1], Y_train_full[1] = xy_split(training_set_full[1])
    X_train_full[2], Y_train_full[2] = xy_split(training_set_full[2])
    return X_train_full, Y_train_full, X_test, Y_test


# Function to obtain the model auc or acc on test set
def get_model_auc_test(model, X_train, Y_train, acc=False):
    global X_test, Y_test
    model.fit(X_train, Y_train)
    Y_pre = model.predict(X_test)
    if acc:
        return accuracy_score(Y_pre, Y_test)
    return roc_auc_score(Y_pre, Y_test)


# Function to obtain the model auc or acc
def get_model_auc(model, X_train, Y_train, X_test, Y_test, acc=False, prob=False, fit=True):
    if fit:
        model.fit(X_train, Y_train)
    Y_pre = model.predict(X_test)

    if acc:
        score = accuracy_score(Y_pre, Y_test)
    else:
        score = roc_auc_score(Y_pre, Y_test)

    if prob:
        Y_prob = model.predict_proba(X_test)
        return score, Y_prob, model
    return score, Y_pre, model


# Function to traversal search the best length of the dataset and the best model
def model_selection(current_date):
    global X_train_full, Y_train_full, dev_set_index_full
    if os.path.exists('Database/result/model_selection/' + 'model' + '_' + current_date + '.pkl'):
        index_train, index_model = read_from_file('model' + '_' + current_date, 'Database/result/model_selection')
        print('Best Params: ', index_train, ' ', index_model)
        return xgb_model[index_model], index_train + 1, X_train_full[index_train], Y_train_full[index_train], dev_set_index_full[index_train]

    best_auc = -1
    for i in range(1):
        for j in range(3):
            model = xgb_model[j]
            X_train_current = X_train_full[i]
            Y_train_current = Y_train_full[i]
            dev_set_index = dev_set_index_full[i]

            ps = PredefinedSplit(test_fold=dev_set_index)
            auc = cross_validate(model, X_train_current, Y_train_current, scoring='roc_auc', cv=ps, n_jobs=-1, return_train_score=False)['test_score'].mean()

            acc = get_model_auc_test(model, X_train_current, Y_train_current)
            print('%.3f  %.3f  %d  %d' % (auc, acc, i, j))
            if auc > best_auc:
                best_auc = auc
                index_model = j
                index_train = i

    print('Best Params: ', index_train, ' ', index_model)

    save_to_file([index_train, index_model], 'model' + '_' + current_date, 'Database/result/model_selection')
    return xgb_model[index_model], index_train + 1, X_train_full[index_train], Y_train_full[index_train], dev_set_index_full[index_train]


# Copy the model as model_copy
def model_copy(model):
    model_copy = XGBClassifier()
    model_copy.learning_rate = model.learning_rate
    model_copy.n_estimators = model.n_estimators
    model_copy.max_depth = model.max_depth
    model_copy.min_child_weight = model.min_child_weight
    model_copy.gamma = model.gamma
    model_copy.subsample = model.subsample
    model_copy.colsample_bytree = model.colsample_bytree
    model_copy.reg_alpha = model.reg_alpha
    model_copy.kwargs['reg_lamda'] = model.kwargs['reg_lamda']

    model_copy.objective = model.objective
    model_copy.nthread = model.nthread
    model_copy.scale_pos_weight = model.scale_pos_weight
    model_copy.seed = model.seed
    model_copy.kwargs['tree_method'] = model.kwargs['tree_method']
    # model_copy.predictor=model.predictor

    return model_copy


# Plot the model auc figure on the test set by date
def plot_one_dim_50(x_axis, y_axis_group, label=['auc'], x_label='date', y_label='auc', save_path=None, font_size=35):
    plt.figure(figsize=(0.5 * len(x_axis), 10))

    plt.xticks(fontsize=font_size, rotation=50)
    plt.yticks(fontsize=font_size)

    x_major_locator = MultipleLocator(6)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    x_axis = np.array(x_axis)
    y_axis_group = [np.array(y_axis) for y_axis in y_axis_group]

    for index, value in enumerate(y_axis_group):
        plt.plot(x_axis, value, label=label[index], linewidth=6)

    plt.plot(x_axis, [50.0] * len(x_axis), label='50%', linewidth=3, linestyle='--')
    plt.legend(fontsize=font_size * 1.2)
    plt.xlabel(x_label, fontsize=font_size * 1.2)
    plt.ylabel(y_label, fontsize=font_size * 1.2)

    if save_path != None:
        plt.savefig(save_path + '.png')
    plt.show()


# Plot the one dimension figure on the test set by date
def plot_one_dim(x_axis, y_axis, label='auc', x_label='auc', y_label='auc', save_path=None):
    if not os.path.exists('Database/result/feature_num'):
        os.makedirs('Database/result/feature_num')
    x_axis = np.array(x_axis)
    y_axis = np.array(y_axis)

    plt.plot(x_axis, y_axis, label=label)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_path != None:
        plt.savefig(save_path + '.png')

    plt.show()


# Feature selection to determine the optimal number of features
def feature_selection(X_train_complete, Y_train_complete):
    global X_test, Y_test, model, year, interval, current_date, drop

    if os.path.exists('Database/result/feature_num/' + 'thresh_' + current_date + '.pkl'):
        from PIL import Image
        im = Image.open('Database/result/feature_num/feature_num_auc_' + current_date + '.png')
        im.show()

        best_thresh_auc, best_auc_test = read_from_file('thresh_' + current_date, 'Database/result/feature_num')
        print('Best_Thresh_Auc: ', best_thresh_auc)
        print('Thresh Acc on Test: ', best_auc_test)
        return best_thresh_auc, best_auc_test

    training_set, dev_set, test_set = get_dataset_bydate_dev(current_date, date_list, interval=interval, dev_set=True,
                                                             year=1, drop=drop)
    X_train, Y_train = xy_split(training_set)
    X_dev, Y_dev = xy_split(dev_set)

    x_axis, y_axis_auc = [], []
    best_auc, best_acc, best_thresh_auc, best_thresh_acc = -1, -1, -1, -1
    model.fit(X_train_complete, Y_train_complete)
    model_selection = model_copy(model)

    from numpy import sort
    from sklearn.feature_selection import SelectFromModel
    # Sort the feature_importances_ by importance level
    thresholds = sort(model.feature_importances_)
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        select_X_dev = selection.transform(X_dev)
        model_selection.fit(select_X_train, Y_train)
        Y_pre = model_selection.predict(select_X_dev)

        auc = roc_auc_score(Y_dev, Y_pre)

        if auc > best_auc:
            best_auc = auc
            best_thresh_auc = thresh

        x_axis.append(select_X_train.shape[1])
        y_axis_auc.append(auc * 100)
        if select_X_train.shape[1] % 10 == 0:
            print("Thresh=%.4f, n=%d, Auc: %.2f%%" % (
                thresh, select_X_train.shape[1], auc * 100.0))
    print_time(start_time)

    # Plot the auc figure on the dev set
    plot_one_dim(x_axis, y_axis_auc, label='auc', x_label='feature_num', y_label='auc',
                 save_path='Database/result/feature_num/feature_num_auc_' + current_date)

    print('Best_Thresh_Auc: ', best_thresh_auc)
    selection = SelectFromModel(model, threshold=best_thresh_auc, prefit=True)
    select_X_train = selection.transform(X_train_complete)
    model_selection.fit(select_X_train, Y_train_complete)
    select_X_test = selection.transform(X_test)
    Y_pre = model_selection.predict(select_X_test)
    best_auc_test = accuracy_score(Y_test, Y_pre)
    print('Thresh Acc on Test: ', best_auc_test)

    save_to_file([best_thresh_auc, best_auc_test], 'thresh_' + current_date, 'Database/result/feature_num')

    return best_thresh_auc, best_auc_test


# Function to grid search the optimal parameters for the chosen best model
def GridSearchCV_on_Dev(current_date, year, randomize=True, overwrite=False):
    global X_train, Y_train, dev_set_index, model
    if (overwrite == False) and (
            os.path.exists('Database/result/model/' + 'model' + '_' + current_date + '_' + str(year) + '.pkl')):
        grid_search_model, year = read_from_file('model' + '_' + current_date + '_' + str(year),
                                                 'Database/result/model')
        print('Best Params: ', grid_search_model.best_params_)
        print('Best Auc: ', grid_search_model.best_score_)
        model = grid_search_model.best_estimator_

        # model=model_ungpu(model)

        # print('Acc on Test: ', get_model_auc_test(model, X_train, Y_train, acc=True))
        return model, year

    # Implement the PredefinedSplit to grid search on the dev set
    ps = PredefinedSplit(test_fold=dev_set_index)

    para_list = {
        'learning_rate': [0.04, 0.1, 0.24],
        'subsample': [i / 100.0 for i in range(75, 91, 5)],
        'colsample_bytree': [i / 100.0 for i in range(75, 91, 5)],
    }

    if randomize:
        grid_search_model = RandomizedSearchCV(estimator=model,
                                               param_distributions=para_list, n_iter=24, scoring='roc_auc', n_jobs=-1, iid=False, cv=ps, verbose=1)
    else:
        grid_search_model = GridSearchCV(estimator=model,
                                         param_grid=para_list, scoring='roc_auc', n_jobs=-1, iid=False, cv=ps, verbose=1)

    grid_search_model.fit(X_train, Y_train)

    print('Best Params: ', grid_search_model.best_params_)
    print('Best Auc: ', grid_search_model.best_score_)
    model = grid_search_model.best_estimator_
    # print('Acc on Test: ', get_model_auc_test(model, X_train, Y_train, acc=True))

    save_to_file([grid_search_model, year], 'model' + '_' + current_date + '_' + str(year), 'Database/result/model',
                 overwrite=overwrite)

    return model, year


if __name__ == '__main__':
    start_time_total = time.time()
    sum = 0
    pro = authorization_jq(1)

    drop = True
    interval = 20
    start_date = '2007-01-01'
    end_date = '2020-07-15'
    base_date = '2015-12-31'
    date_list = get_date_list(start_date, end_date, base_date, interval=interval)
    date_list_selected = list_slice(date_list, '2010-01-25', '2020-06-10')

    # xgb_model[0],xgb_model[1],xgb_model[2]=model_ungpu(xgb_model[0]),model_ungpu(xgb_model[1]),model_ungpu(xgb_model[2])

    '''
    # Execute from the middle to both sides
    middle_date='2015-06-08'
    date_index=date_list_selected.index(middle_date)
    #for current_date in date_list_selected[date_index::-1]:
    for current_date in date_list_selected[date_index+1:]:
    '''

    for current_date in date_list_selected:
        # for current_date in date_list_selected[::-1]:
        start_time = time.time()
        print(current_date + '\n')

        training_set_full, dev_set_index_full, test_set, = get_datasetfull_bydate(current_date, date_list, interval=interval)
        X_train_full, Y_train_full, X_test, Y_test = xy_split_all()

        # Model selection
        model, year, X_train, Y_train, dev_set_index = model_selection(current_date)
        print_time(start_time)

        # year,model,X_train,Y_train,dev_set_index=1,xgb_model[0],X_train_full[year-1],Y_train_full[year-1],dev_set_index_full[year-1]

        model, year = GridSearchCV_on_Dev(current_date, year, randomize=True)
        #model=model_ungpu(model)
        print_time(start_time)

        best_thresh_auc, best_auc_test = feature_selection(X_train, Y_train)

        acc_test = get_model_auc_test(model, X_train, Y_train, acc=True)
        print('\nFinal Acc on Test: ', acc_test)

        print_time(start_time)
        print_time(start_time_total, 'total ', wrap_above=False)

        #save_to_google_drive()

    #save_to_google_drive(force_save=True)
    plt.close()