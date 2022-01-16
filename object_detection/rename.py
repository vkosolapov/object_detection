def rename(file_name, string_to_add):
    with open(file_name, 'r') as f:
        file_lines = [''.join([string_to_add, x.strip(), '\n']) for x in f.readlines()]
    with open(file_name, 'w') as f:
        f.writelines(file_lines)


rename("data/AFO/PART_1/PART_1/train.txt", "data/AFO/PART_1/PART_1/images/")
rename("data/AFO/PART_1/PART_1/validation.txt", "data/AFO/PART_1/PART_1/images/")
rename("data/AFO/PART_1/PART_1/test.txt", "data/AFO/PART_1/PART_1/images/")
