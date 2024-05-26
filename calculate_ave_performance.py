import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt


# date_list = [0902_150000_151900, 0902_190000_191900, 0903_150000_151900, 0903_190000_191900, 
#              0924_150000_151900, 0924_190000_191900, 0925_150000_151900, 0925_190000_191900, 
#              1015_150000_151900, 1015_190000_191900]

date_list = ['0902_150000_151900', '0903_150000_151900', '0924_150000_151900', '0925_150000_151900','1015_150000_151900']
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--result_dir', '-f', type=str, help='Directory containing test result.')
    parser.add_argument('--model', '-m', type=str, default='resnet101_ibn_a', help='the name of the pre-trained PyTorch model')
    parser.add_argument('--ensemble', default=False, type=bool, help='Specify whether it is in ensemble mode')
    parser.add_argument('--parameter', '-p', type=str)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--step', type=int)
    parser.add_argument('--cam', type=str)
    parser.add_argument('--time', type=str)
    args = parser.parse_args()


    if args.ensemble:
        save_path = os.path.join(args.result_dir, args.cam)
    else:
        save_path = os.path.join(args.result_dir, args.cam, args.model)
    os.makedirs(save_path, exist_ok=True)
    if args.time == 'm':
        f = open(f'{save_path}/{args.parameter}.txt', 'w')
        date_list = ['0902_150000_151900', '0903_150000_151900', '0924_150000_151900', '0925_150000_151900','1015_150000_151900']
    else:
        f = open(f'{save_path}/night_{args.parameter}.txt', 'w')
        date_list = ['0902_190000_191900', '0903_190000_191900', '0924_190000_191900', '0925_190000_191900','1015_190000_191900']

    max_total = -100
    max_idf1 = -100
    max_mota = -100
    max_i = -100

    y = []
    x = []

    for i in range(args.start, args.end+1, args.step):

        ave_idf1 = 0
        ave_mota = 0
        num = 0
        for date in date_list:
            if args.ensemble:
                file_path = os.path.join(args.result_dir,f'{date}_{args.parameter}_{i}',f'{args.cam}.txt')
            else:
                file_path = os.path.join(args.result_dir,f'{date}_{args.parameter}_{i}',args.model,f'{args.cam}.txt')
            result_file = open(f'{file_path}', 'r')

            line = result_file.readline()
            if line == "":
                continue
            ave_idf1 += float(line.split(',')[0])
            ave_mota += float(line.split(',')[1])
            num += 1
            result_file.close()

        ave_idf1 /= num
        ave_mota /= num
        f.write(f'{args.parameter} {i/100}, AVE IDF1 : {ave_idf1}, AVE MOTA : {ave_mota}, Total Score : {ave_idf1+ave_mota}\n')
        x.append(i/100)
        y.append(ave_idf1+ave_mota)
        if (ave_idf1 + ave_mota) > max_total:
            max_total = ave_idf1 + ave_mota
            max_idf1 = ave_idf1
            max_mota = ave_mota
            max_i = i/100

    f.write(f'Best Parameter : {args.parameter} {max_i}, AVE IDF1 : {max_idf1}, AVE MOTA : {max_mota}, Total Score : {max_total}\n')

    plt.plot(x, y)
    plt.ylabel('Total score')
    plt.xlabel(f'{args.parameter} size')
    if args.time =='m':
        plt.savefig(f'{save_path}/{args.parameter}.jpg')
    else:
        plt.savefig(f'{save_path}/night_{args.parameter}.jpg')
    plt.close()


