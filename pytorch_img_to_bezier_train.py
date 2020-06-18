from pytorch_high_level import *
from pytorch_model import Bezier
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description='IL')
parser.add_argument('--dataset_name', type=str, default="4", help='suffix for the dataset name')


HEIGHT = 240
WIDTH = 320
CHANNELS = 1


net = Bezier(HEIGHT,WIDTH,CHANNELS).to(device)
# optimizer = optim.Adam(net.parameters(),lr = 0.001)
# loss_function = nn.MSELoss()
model_name = 'bezier_1.5.h5'
# checkpoint = torch.load(model_name)
# net = checkpoint['net']
train_log = []
for _ in range(5):
    for i in range(4):
        torch.cuda.empty_cache()
        train_data = np.load('MUSHR_320x240_shuffled_Bezier{}.npy'.format(str(i)),allow_pickle=True)
        X = torch.Tensor([i[0] for i in train_data]).view(-1,CHANNELS,HEIGHT,WIDTH)
        Y = torch.Tensor([i[1] for i in train_data])
        del train_data
        print(i*_)
        train_log = fit(net,X,Y,train_log,optimizer='adam',loss_function='mean_square',validation_set=0.1,BATCH_SIZE=8,EPOCHS=2)
        state = {'net':net}
        torch.save(state,model_name)
        loss_function = None
        optimizer = None
        train_data = None
        X = None
        Y = None
        del train_data
        del X
        del Y
        del loss_function
        del optimizer

a = time.localtime(time.time())
log_file = 'bezier_log_{}_{}_{}_{}_{}.npy'.format(a.tm_year,a.tm_mon,a.tm_mday,a.tm_hour,a.tm_min)
np.save(log_file,np.array(train_log))
# TODO: add a snapshot step to autosave model.