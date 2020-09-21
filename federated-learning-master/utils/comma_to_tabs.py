'''
A script that replaces commas with tabs in a file
'''

in_filename = input("Please input the file name:")
out_filename = "out_"+in_filename

with open(in_filename,"r") as fin:
    with open(out_filename,"w+") as fout:
        for line in fin:
            fout.write(line.replace(',','\t'))

print("Output File:",out_filename)
