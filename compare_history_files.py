from tensorboardX import SummaryWriter
import cPickle
writer = SummaryWriter()

history_folders = ['Pretrained_Filters_Max_Pooling_Run1_wd_5/','Pretrained_Filters_Min_Pooling_Run1_wd_5/','Pretrained_Filters_None_Pooling_Run1_wd_5/']

histories = {}
for history_path in history_folders:
    histories[history_path] = cPickle.load(open(history_path+'history.txt','r'))

for run,history in histories.iteritems():
    for layer_index,values in history.iteritems():
        for key,value in values.iteritems():
            for i,current_point in enumerate(value):
                 writer.add_scalar('data/'+run+'_'+layer_index+'_'+key, current_point, i)
