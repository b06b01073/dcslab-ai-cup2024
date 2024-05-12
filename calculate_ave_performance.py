import os
from argparse import ArgumentParser

# date_list = [0902_150000_151900, 0902_190000_191900, 0903_150000_151900, 0903_190000_191900, 
#              0924_150000_151900, 0924_190000_191900, 0925_150000_151900, 0925_190000_191900, 
#              1015_150000_151900, 1015_190000_191900]

date_list = ['0902_150000_151900', '0903_150000_151900', '0924_150000_151900', '0925_150000_151900', '1015_150000_151900']
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--result_dir', '-f', type=str, help='Directory containing test result.')
    parser.add_argument('--parameter', '-p', type=str)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--step', type=int)
    parser.add_argument('--cam', type=str)
    args = parser.parse_args()


    save_path = os.path.join(args.result_dir, args.cam)
    os.makedirs(save_path, exist_ok=True)
    f = open(f'{save_path}/{args.parameter}.txt', 'w')
    for i in range(args.start, args.end+1, args.step):

        ave_idf1 = 0
        ave_mota = 0
        num = 0
        for date in date_list:
            file_path = os.path.join(args.result_dir,f'{date}_{args.parameter}_{i}')
            result_file = open(f'{file_path}/{args.cam}.txt', 'r')

            line = result_file.readline()
            if line == "":
                continue
            ave_idf1 += float(line.split(',')[0])
            ave_mota += float(line.split(',')[1])
            num += 1
            result_file.close()

        ave_idf1 /= num
        ave_mota /= num
        f.write(f'{args.parameter} {i/100}, AVE IDF1 : {ave_idf1}, AVE MOTA : {ave_mota}\n')
       