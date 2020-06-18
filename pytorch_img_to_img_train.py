from pytorch_high_level_img_to_img import *
from pytorch_model import trajectory
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description='IL')
parser.add_argument('--dataset_name', type=str, default="4", help='suffix for the dataset name')


HEIGHT = 240
WIDTH = 320
CHANNELS = 1


net = trajectory(HEIGHT,WIDTH,CHANNELS).to(device)
model_name = 'trajectory_warp.h5'
# checkpoint = torch.load(model_name)
# net = checkpoint['net']
print("model loaded")
train_log = []
for i in range(4):
    torch.cuda.empty_cache()
    train_data = np.load('MUSHR_320x240_shuffled_{}.npy'.format(str(i)),allow_pickle=True)
    X = np.array([i[0] for i in train_data])
    Y = np.array([i[1] for i in train_data])
    print("data loaded")
    train_data = None
    del train_data
    train_log = fit(net,X,Y,train_log,optimizer='adam',loss_function='mean_square',validation_set=0.1,BATCH_SIZE=16,EPOCHS=10)
    state = {'net':net}
    torch.save(state,'trajectory_1.h5')
    loss_function = None
    optimizer = None
    X = None
    Y = None
    del X
    del Y
    del loss_function
    del optimizer

# a = time.localtime(time.time())
# log_file = 'trajectory_log_{}_{}_{}_{}_{}.npy'.format(a.tm_year,a.tm_mon,a.tm_mday,a.tm_hour,a.tm_min)
# np.save(log_file,np.array(train_log))
# TODO: add a snapshot step to autosave model.