import csv

filename = "annotation.csv"
features = ['sample_filename', 'folder', 'classID', 'className']
genres = ['country', 'pop', 'rock']  # TODO: add more genres with 116 songs in each

with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(features)

    # writing the data rows
    for i in range(3):
        for j in range(116):  # we have 115 songs for each genre
            sample_filename = f"{genres[i]}_{j}.mp3"
            folder = f"{i + 1}"
            classID = f"{i}"
            row = [sample_filename, folder, classID, genres[i]]
            print(row, '\n')
            csvwriter.writerow(row)
