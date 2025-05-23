from torch.utils.data import Dataset
from fluidResconstruction.render import  *

class TestScalarflowProcDataset(Dataset):
    def __init__(self, modelConfig, path, imagetransform=None,start_frame = 0, end_frame = 150):
        self.modelConfig = modelConfig
        self.size = modelConfig['den_size']
        self.velsize = modelConfig['vel_size']
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.ITEM = self.end_frame - self.start_frame + 1

        self.denpath = path
        self.imagetransform = imagetransform



    def resize3D(self, x, size):
        resized_data = F.interpolate(x, size=size, mode='trilinear',
                                     align_corners=False)
        return resized_data
    def bin2tensor(self, index):
        index = index + self.start_frame
        file = [os.path.join(self.denpath, f) for f in os.listdir(self.denpath) if f.endswith('.npz') and (f'imgs_{index:06d}' in f )]
        if not file:
            raise FileNotFoundError(f"No file found for imgs_{index:06d} in {self.denpath}")


        data = np.load(file[0])['data']
        data = torch.from_numpy(data)
        front = data[0:1,:,:,0] #[1920, 1080]
        side = data[3:4,:,:,0]
        front = torch.rot90(front, k=2, dims=(1, 2))
        side = torch.rot90(side, k=2, dims=(1, 2))
        top, left, bottom, right = 40, 115, 1040, 1895
        front = front[:, left:right, top:bottom]  # [1780, 1000]
        top, left, bottom, right = 40, 105, 1040, 1885
        side = side[:, left:right, top:bottom]  # [1780, 1000]
        return front, side, index


    def __getitem__(self, index):
        # num = index + 150 * 89

        front,side,num = self.bin2tensor(index)


        if self.imagetransform:
            front = self.imagetransform(front)
            side = self.imagetransform(side)


        den = torch.zeros(1,self.size[0], self.size[1], self.size[2])
        vel = torch.zeros(self.velsize)
        # print(vel.shape)
        # print(front.max())


        return {'front_view': front*2,
                'side_view': side*2,
                'den': den,
                'num': str(f'{num:06d}'),
                'vel': vel,
                }


    def __len__(self):
        return self.ITEM
class TestSynthesisDataset(Dataset):
    def __init__(self, modelConfig, denpath,velpath, imagetransform=None, dentransform=None,start_frame = 0, end_frame = 149,scene_num=81):
        self.modelConfig = modelConfig
        self.size = modelConfig['den_size']
        self.velsize = modelConfig['vel_size']
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.ITEM = self.end_frame - self.start_frame + 1

        self.denpath = denpath
        self.velpath = velpath
        self.imagetransform = imagetransform
        self.dentransform = dentransform
        self.scene_num = scene_num

    def bin2tensor(self, index):
        index = index + 150 * self.scene_num
        index = index + self.start_frame
        file = [os.path.join(self.denpath, f) for f in os.listdir(self.denpath) if f.endswith('.bin') and (f'density{index:06d}' in f )]
        if not file:
            raise FileNotFoundError(f"No file found for density_{index:06d} in {self.denpath}")

        if index > 150 * self.scene_num:
            velfile = [os.path.join(self.velpath, f) for f in os.listdir(self.velpath) if
                f.endswith('.bin') and (f'velocity{index-1:06d}' in f)]
            if not velfile:
                raise FileNotFoundError(f"No file found for velocity_{index - 1:06d} in {self.velpath}")
            velData = np.fromfile(velfile[0], dtype=np.float32)
            velData = velData.reshape(self.velsize)
            vel = torch.from_numpy(velData)
        else:
            vel = 0

        data = np.fromfile(file[0], dtype=np.float32)
        data = data.reshape(self.size)
        front = np.mean(data, axis=0)
        front = np.expand_dims(front, axis=0)
        side = np.mean(data, axis=2)
        side = np.expand_dims(side, axis=0)

        front_view = torch.from_numpy(front)
        side_view = torch.from_numpy(side)
        data = np.expand_dims(data, axis=0)
        data = torch.from_numpy(data)


        return data, front_view, side_view, vel, index


    def __getitem__(self, index):
        # num = index + 150 * 89

        den,front,side,vel,num = self.bin2tensor(index)


        if self.imagetransform:
            front = self.imagetransform(front)
            side = self.imagetransform(side)

        if self.dentransform:
            den = self.dentransform(den)

        # print(vel.shape)


        return {'front_view': front,
                'side_view': side,
                'den': den,
                'num': str(f'{num:06d}'),
                'vel': vel,
                'isScalar': False,
                }


    def __len__(self):
        return self.ITEM



