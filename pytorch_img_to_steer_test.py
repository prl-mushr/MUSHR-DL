from pytorch_high_level import *
# from pytorch_model import Net


HEIGHT = 240
WIDTH = 320
CHANNELS = 1


# net = Net(HEIGHT,WIDTH,CHANNELS).to(device)
# optimizer = optim.Adam(net.parameters(),lr = 0.001)
# loss_function = nn.MSELoss()
checkpoint = torch.load('steering.h5')
model = checkpoint['net']
model = model.to(device)
# train_log = []
for i in range(1):
    # optimizer = optim.Adam(net.parameters(),lr = 0.001)
    # loss_function = nn.MSELoss()
    # torch.cuda.empty_cache()
    train_data = np.load('MUSHR_320x240_shuffled_steering_{}.npy'.format(str(i)),allow_pickle=True)
    X = ([i[0] for i in train_data])
    Y = ([i[1] for i in train_data])
    with torch.no_grad():
        for j in range(0,len(X),10):
            img = np.array(X[j],dtype=np.float32)
            time.sleep(0.001)
            img = torch.Tensor(img).view(-1,1,240,320).to(device)
            # print(img.dtype)
            now = time.time()
            output = model(img)[0]
            delta = time.time()-now
            print(delta*1000)
            output = float(output[0] - output[1])
            GT = Y[j]
            GT = GT[0] - GT[1]
            error = np.fabs(GT-output)
            print(error)
            # cv2.imshow('window',X[j])
            # cv2.waitKey(1)
            # time.sleep(1)
    train_data = None
    X = None
    Y = None
    del train_data
    del X
    del Y