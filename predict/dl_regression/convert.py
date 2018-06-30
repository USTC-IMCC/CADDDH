import csv
with open('file.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, dialect='excel')
    with open('label.txt', 'rb') as filein:
        for line in filein:
            line_list = line.strip('\n').split(' ')
            spamwriter.writerow(line_list)
