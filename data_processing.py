####################################
# Assignment: 3                    #
# Student Name: Shuangquan Zheng   #
####################################
import csv
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
 

def load_metrics(fp):
    """
    Use csv reader first to import data (delimiter is ',' and quotechar is
    set to double quote then append them to list then turn it into numpy array.
    """
    with open(fp) as csvfile:
        read = csv.reader(csvfile, delimiter=",", quotechar='"')
        lst = []
        for i in read:
            new_row = i[0:2] + i[7:-1]
            lst.append(new_row)
        data = np.array(lst)
    return data


def unstructured_to_structured(data, indexes):
    """
    Convert header to list first then delete it in the numpy array and loop
    through header and if the index is in header then make tuple of the name
    of colmn and types <U30 else float 64 and append it to dtype_list then
    use loop to convert the numpy to tuple and then finally convert everything to
    updated type  with dtype=dtype_lst.
    """
    header = data[0].tolist()
    dtype_lst = []
    for i in range(len(header)):
        if i in indexes:
            dtype_lst.append(tuple([header[i], "<U30"]))
        else:
            dtype_lst.append(tuple([header[i], "float64"]))
    updated = np.delete(data, 0, 0)
    tuple_lst = []
    for i in updated:
        tuple_lst.append(tuple(i))
    updated = np.array(tuple_lst, dtype=dtype_lst)
    return updated


def converting_timestamps(array):
    """
    Use month dict for faster conversion of month(abbr to number) then use
    format to make the string and change the value through indexing if the number
    of column is less or equal to 1 then just i.split not i[0].
    """
    row = 0
    data = array
    month_dict = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
                  "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
                  "Sept": "09", "Oct": "10", "Nov": "11", "Dec": "12"}
    for i in array:
        if len(data[0][0]) <= 1:
            string_lst = i.split()
            data[row] = np.array("{}-{}-{} {}".format(string_lst[5], month_dict[string_lst[1]],
                                                      string_lst[2], string_lst[3]))
            row += 1
        else:
            string_lst = i[0].split()
            data[row][0] = np.array("{}-{}-{} {}".format(string_lst[5],
                                                         month_dict[string_lst[1]], string_lst[2], string_lst[3]))
            row += 1
    return data


def replace_nan(data):
    """
    Make a numpy index list then loop each of it for its respective colmn
    and get its mean and use where function to replace all values that is
    corrupted.
    """
    lst_ind = np.array(['valence_intensity', 'anger_intensity',
                        'fear_intensity', 'sadness_intensity', 'joy_intensity'])
    for i in lst_ind:
        native = data[:][i]
        avg = np.nanmean(native)
        data[:][i] = np.where(np.isnan(native), avg, native)
    return data


def boxplot_data(data, output_name="output.png"):
    """
    Draw plot as instructed and use plt.
    """
    xlabels = ['Valence', 'Anger', 'Fear', 'Sadness', 'Joy']
    data_to_plt = [
        data[:]["valence_intensity"], data[:]['anger_intensity'],
        data[:]['fear_intensity'], data[:]['sadness_intensity'],
        data[:]['joy_intensity']
    ]
    b_dict = {'patch_artist': True,
              'medianprops': dict(linestyle='-', linewidth=1, color='k')}
    plt.figure(figsize=(10, 7))
    boxs = plt.boxplot(data_to_plt, labels=xlabels, **b_dict)
    plt.title('Distribution of Sentiment')
    plt.grid(axis='y')
    plt.ylabel('Values')
    plt.xlabel('Sentiment')
    colors = ['green', 'red', 'purple', 'blue', 'yellow']
    for square, color in zip(boxs['boxes'], colors):
        square.set_facecolor(color)
    plt.plot()
    # Only comment below line when debugging. Uncomment when submitting
    plt.savefig(output_name)


def number_of_outliers(sentiment, lower, upper):
    """
    Use percentile function to get the quartiles and then coun_nonzero which
    will count any true (which will be the outliers).
    """
    upper_quartile = np.percentile(sentiment, upper)
    lower_quartile = np.percentile(sentiment, lower)
    lower_outlier = np.count_nonzero(sentiment <= lower_quartile)
    higher_outlier = np.count_nonzero(sentiment >= upper_quartile)
    total_outlier = lower_outlier + higher_outlier
    return total_outlier


def convert_to_df(data):
    """
    Use pd dataframe to convert numpy to dataframe and return it.
    """
    ans = pd.DataFrame(data)
    return ans


def load_tweets(fp):
    """
    TSV extensive by google definition is Tab-separated values so seprater
    is set to tab isntead of newline.
    """
    ans = pd.read_csv(fp, sep='\t')
    return ans


def merge_dataframes(df_metrics, df_tweets):
    """
    Uses .rename to change the column name to twwet_ID for df_tweets then
    change mertics colmn type to tweet_ID to float64 then to int64 and then
    merge two dataframe.
    """
    df_tweets = df_tweets.rename(columns={'id': 'tweet_ID'})
    df_tweets[['tweet_ID']] = df_tweets[['tweet_ID']].astype('int64')
    df_metrics[['tweet_ID']] = df_metrics[['tweet_ID']].astype(
        "float64").astype('int64')
    ans = df_tweets.join(
        df_metrics.set_index('tweet_ID'), on='tweet_ID', how='inner').dropna()
    return ans


def plot_timeperiod(df_merged, from_date, to_date, output_name="output.png"):
    """
    Draw only the selected data(from_date - to_date) and then set its
    color to green,red,purple,blue and yellow. First need to change
    dtype of the created_at column to datetime then sort it and finally
    use condition which will be used in loc to filter the data to only
    plot data wanted.
    """
    lst = ['valence_intensity', 'anger_intensity',
           'fear_intensity', 'sadness_intensity', 'joy_intensity']
    df_merged[['created_at']] = pd.to_datetime(
        df_merged.loc[:, 'created_at'], format='%Y-%m-%d %H:%M:%S')
    df_merged = df_merged.sort_values(
        by=['created_at'])
    from_date = pd.to_datetime(
        from_date, format='%Y-%m-%d %H:%M:%S')
    to_date = pd.to_datetime(
        to_date, format='%Y-%m-%d %H:%M:%S')
    color = {'valence_intensity': "green", 'anger_intensity': "red",
             'fear_intensity': "purple", 'sadness_intensity': "blue",
             'joy_intensity': "yellow"}
    df = df_merged.loc[((df_merged['created_at'] > from_date)
                        & (df_merged['created_at'] < to_date))]
    df.plot(x="created_at", y=['valence_intensity', 'anger_intensity',
                               'fear_intensity', 'sadness_intensity',
                               'joy_intensity'],
            figsize=(15, 8), color=color)
    # Only comment below line when debugging. Uncomment when submitting
    plt.savefig(output_name)


def get_top_n_words(column, n):
    """
    retrieves the top n words and their frequencies from given data
    """
    frequencies = Counter()
    column.str.lower().str.split().apply(frequencies.update)
    return frequencies.most_common(n)


def plot_frequency(word_frequency, n, output_name="output.png"):
    # partially completed for you, complete the rest according to the instructions.
    """
    Draw horiztonal bar graph with gradient color
    """
    # setting up plot variables
    words = tuple(zip(*word_frequency))[0]
    frequencies = tuple(zip(*word_frequency))[1]
    y_pos = np.arange(len(words))
    fig, ax = plt.subplots(figsize=(15, 10))
    # set up color spectrum
    colors = [
        "red", "orange", "yellow", "green", "blue", "indigo",
        "violet"
    ]
    rvb = mcolors.LinearSegmentedColormap.from_list("", colors)
    nlist = np.arange(n).astype(float)
    ax.barh(y_pos, frequencies, align='center', color=rvb(nlist/n))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_title("Word Frequency: Top {}".format(n))
    # Only comment below line when debugging. Uncomment when submitting
    plt.savefig(output_name)
