import os


def createAnnotation(folder_path, input_file_name, output_file_name):
    # Specify the folder path
    folder_path = folder_path

    # List all files in the folder
    files = os.listdir(folder_path)

    lines = []

    # Process each file
    for file_name in files:
        songid = 0
        print(file_name)
        # Split the file name by "_"
        file_parts = file_name.split('_')
        # Get the first half of the split
        first_half = file_parts[0]
        # Print or use the first half as needed
        print(first_half)
        if first_half == "country":
            songid = 0
        elif first_half == "pop":
            songid = 1
        elif first_half == "rock":
            songid = 2
        elif first_half == "jazz":
            songid = 3
        elif first_half == "metal":
            songid = 4

        lines.append(file_name + "," + first_half + "," + str(songid))



    print(lines)


    # Specify the file path
    #file_path = "music-analysis-annotation.csv"
    file_path = output_file_name

    # Open the file in write mode ('w' mode)
    with open(file_path, 'w') as file:
        # Write each line to the file
        for line in lines:
            file.write(line + '\n')

    file.close()

createAnnotation("train", "country_0.mp3", "test_train.csv")

print("Lines have been written to the file successfully!")
