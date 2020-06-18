from pytorch_high_level import *
from pytorch_model import Net
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

HEIGHT = 240
WIDTH = 320
CHANNELS = 1


net = Net(HEIGHT,WIDTH,CHANNELS).to(device)
# optimizer = optim.Adam(net.parameters(),lr = 0.001)
# loss_function = nn.MSELoss()
# checkpoint = torch.load('steering.h5')
# net = checkpoint['net']
train_log = []
for i in range(4):
    torch.cuda.empty_cache()
    train_data = np.load('MUSHR_320x240_shuffled_steering_{}.npy'.format(str(3)),allow_pickle=True)
    X = torch.Tensor([i[0] for i in train_data]).view(-1,CHANNELS,HEIGHT,WIDTH)
    Y = torch.Tensor([i[1] for i in train_data])
    del train_data
    train_log = fit(net,X,Y,train_log,optimizer='adam',loss_function='mean_square',validation_set=0.1,BATCH_SIZE=8,EPOCHS=10)
    state = {'net':net}
    torch.save(state,'steering.h5')
    loss_function = None
    optimizer = None
    X = None
    Y = None
    del X
    del Y
    del loss_function
    del optimizer

a = time.localtime(time.time())
log_file = 'steering_log_{}_{}_{}_{}_{}.npy'.format(a.tm_year,a.tm_mon,a.tm_mday,a.tm_hour,a.tm_min)
np.save(log_file,np.array(train_log))
# TODO: add a snapshot step to autosave model.