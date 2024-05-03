from pathlib import Path
filename = "annotation.txt"
features = ['sample_filename', 'folder', 'classID', 'classLabel']
genres = ['country', 'pop', 'rock']  # TODO: add more genres with 116 songs in each

file = open(filename, "w")
# writing the data rows
file.write("sample_filename,folder,classID,className\n")
for i in range(3):
    for j in range(116):  # we have 115 songs for each genre
        sample_filename = f"{genres[i]}_{j}.mp3"
        folder = f"{i + 1}"
        classID = f"{i}"
        file.write(sample_filename + "," + folder + "," + classID + "," + genres[i] + "\n")
file.close()
print(Path.cwd())
