import csv

def getList(name):
    '''Takes file name as parameter. Returns list of data in the CSV file as float values'''
    data = []
    f = open(name, 'rb')
    reader = csv.reader(f)
    for row in reader:
        data.append(float(row[0]))
    f.close()
    return data
