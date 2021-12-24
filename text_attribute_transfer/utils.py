import argparse
import time

import torch
import matplotlib.pyplot as plt
import os

from data import calc_bleu

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")
    
def add_log(args, ss):
    now_time = time.strftime("[%Y-%m-%d %H:%M:%S]: ", time.localtime())
    print(now_time + ss)
    with open(args.log_file, 'a') as f:
        f.write(now_time + str(ss) + '\n')

def add_output(args, ss):
    with open(args.output_file, 'a') as f:
        f.write(str(ss) + '\n')

def write_text_z_in_file(args, text_z_prime) :
    with open(args.output_file, 'w') as f:
        keys = list(text_z_prime.keys())
        k1 = keys.index("source")
        k2 = keys.index("before")
        k3 = keys.index("after")
        K_ =  len(keys)
        L = len(text_z_prime[keys[0]])
        for i in range(L) :
            item = [text_z_prime[k][i] for k in keys]
            for j in range(len(item[0])) :
                f.writelines(["%s : %s\n"% (keys[k], item[k][j]) for k in range(K_)])
                source = item[k1][j].split(" ")
                before = item[k2][j].split(" ")
                after = item[k3][j].split(" ")
                b1 = round(calc_bleu(source, before), 4)
                b2 = round(calc_bleu(source, after), 4)
                b3 = round(calc_bleu(before, after), 4)
                f.writelines(["bleu : source vs before = %s, source vs after = %s, before vs after = %s\n"%(b1, b2, b3)])
                f.write("\n")
                
def plot_all_scores(scores=None, from_path="", prefix = ['train', 'eval'], 
                    to_plot = ['ae_loss', 'ae_acc', 'ae_ppl', 'clf_loss', 'clf_acc']) :
    assert scores is not None or os.path.isfile(from_path)
    if scores is None :
        scores = torch.load(from_path)
        if "all_scores" in scores :
            scores = scores["all_scores"]

    suptitle=""
    k = 0
    if True :
        nrows, ncols = len(to_plot), 1
        fig, ax = plt.subplots(nrows, ncols, sharex=False, figsize = (20, 20))
        fig.suptitle(suptitle)
        for i in range(nrows) :
            name = to_plot[k]
            for p in prefix :
                label = "%s_%s"%(p,name)
                y = [s[label] for s in scores]
                x = list(range(len(y)))
                ax[i].plot(x, y, label=label)
            ax[i].set(xlabel='epoch', ylabel=p)
            ax[i].set_title('%s per epoch'%name)
            ax[i].legend()
            #ax[i].label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.
            k += 1
            if k == len(to_plot) :
                break
    else :
        nrows, ncols = 2, 2
        fig, ax = plt.subplots(nrows, ncols, sharex=False, figsize = (20, 8))
        fig.suptitle(suptitle)
        for i in range(nrows) :
            for j in range(ncols) :
                name = to_plot[k]
                for p in prefix :
                    label = "%s_%s"%(p,name)
                    y = [s[label] for s in scores]
                    x = list(range(len(y)))
                    ax[i][j].plot(x, y, label=label)
                ax[i][j].set(xlabel='epoch', ylabel=p)
                ax[i][j].set_title('%s per epoch'%name)
                ax[i][j].legend()
                #ax[i][j].label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.
                k += 1
                if k == len(to_plot) :
                    break
    plt.show()
    
    
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="discription.")
	parser.add_argument("-p",'--path', type=str)
	args = parser.parse_args()
	path = args.path

	if "deb" in path :
		plot_all_scores(from_path=path, prefix = ['train'])
	else :
		plot_all_scores(from_path=path, prefix = ['train', 'eval'])