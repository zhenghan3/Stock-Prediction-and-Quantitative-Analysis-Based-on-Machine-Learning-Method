# Model analysis
    # Plot model accuracy figure by date
    # Retrieve monthly buy list
    # Plot model importance heat map by date

from matplotlib.pyplot import MultipleLocator

from Model_Selection_XGBoost import *


# Function to plot the model importance heat map by date
def plot_importance_bydate(importance_list, x_axis, y_axis, save_path=None, font_size=12.5):
    x_axis, importance_list = x_axis[::3], importance_list[::3]
    importance_list = np.array(importance_list).T
    if len(importance_list) == 1: importance_list = importance_list.reshape(importance_list.shape[1], 1)
    df = pd.DataFrame(importance_list, index=y_axis, columns=x_axis)
    df = df.reindex(df.sum(axis=1).sort_values(ascending=False).index, axis=0)

    f, ax = plt.subplots(figsize=(20, 18))
    heatmap = sns.heatmap(data=df, cmap=plt.cm.Blues)

    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=font_size * 1.5)

    plt.xticks(fontsize=font_size, rotation=50)
    plt.yticks(fontsize=font_size)

    plt.xlabel('date', fontsize=font_size * 1.5)
    plt.ylabel('factors', fontsize=font_size * 1.5)

    if save_path != None:
        plt.savefig(save_path + '.png')

    plt.show()


if __name__ == '__main__':
    start_time_total = time.time()
    sum = 0
    pro = authorization_jq(1)

    drop=True
    interval = 20
    start_date = '2007-01-01'
    end_date = '2020-07-15'
    base_date = '2015-12-31'
    date_list = get_date_list(start_date, end_date, base_date, interval=interval)
    date_list_selected = list_slice(date_list, '2010-01-25', '2020-06-10')
    # acc_list = read_from_file('acc', 'Database/result')

    year = 1
    acc_list = []
    buy_list = {}
    importance_list = []

    for current_date in date_list_selected:
        # for current_date in date_list_selected[::-1]:
        start_time = time.time()
        print(current_date)

        # Retrieve and split the training set and test set
        training_set, test_set = get_dataset_bydate(current_date, date_list, interval=interval, year=year,drop=drop)
        X_train, Y_train = xy_split(training_set)
        X_test, Y_test = xy_split(test_set)

        # Retrieve the best model of current date from saved file
        grid_search_model = read_from_file('model' + '_' + current_date + '_' + str(year), 'Database/result/model')[0]
        model = grid_search_model.best_estimator_
        #model=model_ungpu(model)

        score, Y_prob, model = get_model_auc(model, X_train, Y_train, X_test, Y_test, acc=True, prob=True, fit=True)

        # Acquire the feature importance sorted list
        importance=model.get_booster().get_score(importance_type="gain")
        importance=[importance.get(f) for f in model.get_booster().feature_names]
        importance=np.array(importance)
        importance=importance/importance.sum()
        importance_list.append(importance)
        #importance_list.append(model.feature_importances_)
        x_axis = [x[2:] for x in date_list_selected[:date_list_selected.index(current_date) + 1]]
        y_axis = X_train.columns.values.tolist()

        acc_list.append(score * 100.0)

        # Take the top N stocks predicted as positive probability to form the buy list
        score, Y_pre, model = get_model_auc(model, X_train, Y_train, X_test, Y_test, acc=True, fit=False)
        Y_prob = pd.DataFrame(np.hstack((Y_prob, Y_test.as_matrix().reshape(len(Y_test), 1))), index=X_test.index)
        Y_prob.sort_values(by=[1], ascending=False, inplace=True)
        buy_list.update({current_date: [list(Y_prob.index)[:20], list(Y_prob.index)[:50], list(Y_prob.index)[:100]]})

    # Plot model accuracy figure by date
    save_to_file(acc_list, 'acc', 'Database/result')
    x_axis = [x[2:] for x in date_list_selected]
    plot_one_dim_50(x_axis, [acc_list], ['test set'], 'date', 'acc', save_path='Database/result/acc')
    # Save buy list
    import json
    jsObj = json.dumps(buy_list)
    fileObject = open('Database/result/buy_list.json', 'w')
    fileObject.write(jsObj)
    fileObject.close()
    # Plot model importance heat map by date
    plot_importance_bydate(importance_list, x_axis, y_axis, save_path='Database/result/importance_bydate')