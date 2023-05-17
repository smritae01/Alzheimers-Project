from data_util import split_csv, read_csv

def testBackgroundCSV():
    csv_file = "../data/datasets/ADNI3/ADNI3.csv"
    output_folder = "../data"
    random_seed = 1051
    data_folder = "../data/datasets/ADNI3"
    lists = read_csv(data_folder+'/ADNI3.csv')
    bgfile, testfile = split_csv(lists[0], lists[1])

    assert len(bgfile) == 13
    

def testTestCSV():

    csv_file = "../data/datasets/ADNI3/ADNI3.csv"
    output_folder = "../data"
    random_seed = 1051
    data_folder = "../data/datasets/ADNI3"
    lists = read_csv(data_folder+'/ADNI3.csv')
    bgfile, testfile = split_csv(lists[0], lists[1])

    assert len(testfile) == 6