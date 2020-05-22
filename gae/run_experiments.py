import os
from datetime import datetime

out_file_name = 'tune_multihead_attn.txt'
itr_times = 10
data_set = 'cora'
model = 'gcn_ae'
num_itr = 200

def grid_search():
    input_drop = [0.4]
    attn_drop = [0.0]
    #feat_drop = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    feat_drop = [0.0]

    with open(out_file_name,'a') as f:
        f.write('{}\n'.format(datetime.now()))
    for ind in input_drop:
        for ad in attn_drop:
            for fd in feat_drop:
                with open(out_file_name,'a') as f:
                    f.write('In_drop rate: {}, attn_drop rate: {}, feat_drop rate: {}, average of {} experiments\n'.format(ind,ad,fd,itr_times)) 
                os.system("python train.py --in_drop {} --attn_drop {} --feat_drop {} --dataset {} --model {} --output_name {} --num_experiments {} --multihead_attn".format(ind,ad,fd,data_set,model,out_file_name,itr_times))

def main():
    grid_search()
    #individual_experiments()

if __name__ == "__main__":
    main()
    