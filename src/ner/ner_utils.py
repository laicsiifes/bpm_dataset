def dump_report(rep_dict: dict, output_file: str):
    if not rep_dict:
        print('\nReport is empty!')
        return
    with open(output_file, 'w') as out_file:
        out_file.write('label;precision;recall;f1-score\n')
        for label in rep_dict.keys():
            if label != 'accuracy':
                out_file.write(f"{label};{rep_dict[label]['precision']};{rep_dict[label]['recall']};"
                               f"{rep_dict[label]['f1-score']}\n")
            else:
                out_file.write(f"{label};;;{rep_dict[label]}\n")
