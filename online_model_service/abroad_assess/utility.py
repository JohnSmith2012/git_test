import csv


def csvread(fname):
    list = []
    with open(fname, 'rt') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            list.append(row)
    return list


def csvreadOneCol(fname, colName):
    list = []
    with open(fname, 'rt') as file:
        print(fname)
        reader = csv.reader(file, delimiter=',')
        col = reader.__next__().index(colName)
        for data in reader:
            list.append(data[col])
            print(data[col])
    return list


def csvwrite(fname, list):
    with open(fname, 'wt') as os:
        writer = csv.writer(os, delimiter=',')
        for item in list:
            writer.writerow(item)