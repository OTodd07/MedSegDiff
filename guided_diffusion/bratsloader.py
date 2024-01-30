import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import SimpleITK as sitk
import torchvision.utils as vutils

train_ids = ['HGG/Brats18_2013_27_1/Brats18_2013_27_1',
            'HGG/Brats18_TCIA01_448_1/Brats18_TCIA01_448_1',
            'HGG/Brats18_CBICA_APZ_1/Brats18_CBICA_APZ_1',
            'HGG/Brats18_TCIA02_430_1/Brats18_TCIA02_430_1',
            'HGG/Brats18_TCIA02_171_1/Brats18_TCIA02_171_1',
            'HGG/Brats18_CBICA_AVG_1/Brats18_CBICA_AVG_1',
            'HGG/Brats18_CBICA_AUR_1/Brats18_CBICA_AUR_1',
            'HGG/Brats18_TCIA08_234_1/Brats18_TCIA08_234_1',
            'HGG/Brats18_2013_21_1/Brats18_2013_21_1',
            'HGG/Brats18_TCIA08_113_1/Brats18_TCIA08_113_1',
            'HGG/Brats18_TCIA02_322_1/Brats18_TCIA02_322_1',
            'HGG/Brats18_TCIA02_208_1/Brats18_TCIA02_208_1',
            'HGG/Brats18_2013_23_1/Brats18_2013_23_1',
            'HGG/Brats18_CBICA_AOP_1/Brats18_CBICA_AOP_1',
            'HGG/Brats18_TCIA02_605_1/Brats18_TCIA02_605_1',
            'HGG/Brats18_TCIA01_425_1/Brats18_TCIA01_425_1',
            'HGG/Brats18_2013_17_1/Brats18_2013_17_1',
            'HGG/Brats18_CBICA_BHM_1/Brats18_CBICA_BHM_1',
            'HGG/Brats18_CBICA_ASH_1/Brats18_CBICA_ASH_1',
            'HGG/Brats18_CBICA_ATV_1/Brats18_CBICA_ATV_1',
            'HGG/Brats18_TCIA01_150_1/Brats18_TCIA01_150_1',
            'HGG/Brats18_CBICA_ATB_1/Brats18_CBICA_ATB_1',
            'HGG/Brats18_TCIA04_149_1/Brats18_TCIA04_149_1',
            'HGG/Brats18_TCIA02_151_1/Brats18_TCIA02_151_1',
            'HGG/Brats18_CBICA_ASU_1/Brats18_CBICA_ASU_1',
            'HGG/Brats18_TCIA02_491_1/Brats18_TCIA02_491_1',
            'HGG/Brats18_TCIA02_608_1/Brats18_TCIA02_608_1',
            'HGG/Brats18_CBICA_APY_1/Brats18_CBICA_APY_1',
            'HGG/Brats18_2013_26_1/Brats18_2013_26_1',
            'HGG/Brats18_CBICA_AXJ_1/Brats18_CBICA_AXJ_1',
            'HGG/Brats18_CBICA_ABN_1/Brats18_CBICA_ABN_1',
            'HGG/Brats18_TCIA02_135_1/Brats18_TCIA02_135_1',
            'HGG/Brats18_TCIA01_378_1/Brats18_TCIA01_378_1',
            'HGG/Brats18_CBICA_AQJ_1/Brats18_CBICA_AQJ_1',
            'HGG/Brats18_CBICA_AZH_1/Brats18_CBICA_AZH_1',
            'HGG/Brats18_TCIA08_280_1/Brats18_TCIA08_280_1',
            'HGG/Brats18_CBICA_AQG_1/Brats18_CBICA_AQG_1',
            'HGG/Brats18_TCIA02_179_1/Brats18_TCIA02_179_1',
            'HGG/Brats18_CBICA_AVJ_1/Brats18_CBICA_AVJ_1',
            'HGG/Brats18_TCIA04_361_1/Brats18_TCIA04_361_1',
            'HGG/Brats18_TCIA03_375_1/Brats18_TCIA03_375_1',
            'HGG/Brats18_TCIA06_332_1/Brats18_TCIA06_332_1',
            'HGG/Brats18_2013_10_1/Brats18_2013_10_1',
            'HGG/Brats18_CBICA_ATX_1/Brats18_CBICA_ATX_1',
            'HGG/Brats18_CBICA_AUQ_1/Brats18_CBICA_AUQ_1',
            'HGG/Brats18_CBICA_AQV_1/Brats18_CBICA_AQV_1',
            'HGG/Brats18_2013_25_1/Brats18_2013_25_1',
            'HGG/Brats18_TCIA02_321_1/Brats18_TCIA02_321_1',
            'HGG/Brats18_CBICA_AUN_1/Brats18_CBICA_AUN_1',
            'HGG/Brats18_2013_11_1/Brats18_2013_11_1',
            'HGG/Brats18_TCIA06_409_1/Brats18_TCIA06_409_1',
            'HGG/Brats18_CBICA_AQA_1/Brats18_CBICA_AQA_1',
            'HGG/Brats18_CBICA_ANP_1/Brats18_CBICA_ANP_1',
            'HGG/Brats18_TCIA02_473_1/Brats18_TCIA02_473_1',
            'HGG/Brats18_TCIA01_131_1/Brats18_TCIA01_131_1',
            'HGG/Brats18_CBICA_AQP_1/Brats18_CBICA_AQP_1',
            'HGG/Brats18_TCIA02_471_1/Brats18_TCIA02_471_1',
            'HGG/Brats18_CBICA_AOH_1/Brats18_CBICA_AOH_1',
            'HGG/Brats18_TCIA02_226_1/Brats18_TCIA02_226_1',
            'HGG/Brats18_CBICA_ATD_1/Brats18_CBICA_ATD_1',
            'HGG/Brats18_CBICA_ASA_1/Brats18_CBICA_ASA_1',
            'HGG/Brats18_CBICA_BHK_1/Brats18_CBICA_BHK_1',
            'HGG/Brats18_TCIA08_242_1/Brats18_TCIA08_242_1',
            'HGG/Brats18_TCIA02_455_1/Brats18_TCIA02_455_1',
            'HGG/Brats18_CBICA_AXO_1/Brats18_CBICA_AXO_1',
            'HGG/Brats18_CBICA_AQR_1/Brats18_CBICA_AQR_1',
            'HGG/Brats18_CBICA_AQD_1/Brats18_CBICA_AQD_1',
            'HGG/Brats18_CBICA_AWH_1/Brats18_CBICA_AWH_1',
            'HGG/Brats18_TCIA01_235_1/Brats18_TCIA01_235_1',
            'HGG/Brats18_CBICA_ARF_1/Brats18_CBICA_ARF_1',
            'HGG/Brats18_TCIA06_211_1/Brats18_TCIA06_211_1',
            'HGG/Brats18_TCIA04_479_1/Brats18_TCIA04_479_1',
            'HGG/Brats18_TCIA08_319_1/Brats18_TCIA08_319_1',
            'HGG/Brats18_TCIA01_221_1/Brats18_TCIA01_221_1',
            'HGG/Brats18_CBICA_AXW_1/Brats18_CBICA_AXW_1',
            'HGG/Brats18_CBICA_ATF_1/Brats18_CBICA_ATF_1',
            'HGG/Brats18_CBICA_ASV_1/Brats18_CBICA_ASV_1',
            'HGG/Brats18_TCIA01_203_1/Brats18_TCIA01_203_1',
            'HGG/Brats18_TCIA02_377_1/Brats18_TCIA02_377_1',
            'HGG/Brats18_TCIA03_296_1/Brats18_TCIA03_296_1',
            'HGG/Brats18_TCIA04_343_1/Brats18_TCIA04_343_1',
            'HGG/Brats18_TCIA08_162_1/Brats18_TCIA08_162_1',
            'HGG/Brats18_TCIA06_372_1/Brats18_TCIA06_372_1',
            'HGG/Brats18_2013_18_1/Brats18_2013_18_1',
            'HGG/Brats18_CBICA_ASK_1/Brats18_CBICA_ASK_1',
            'HGG/Brats18_TCIA03_265_1/Brats18_TCIA03_265_1',
            'HGG/Brats18_TCIA04_437_1/Brats18_TCIA04_437_1',
            'HGG/Brats18_CBICA_ALU_1/Brats18_CBICA_ALU_1',
            'HGG/Brats18_TCIA02_290_1/Brats18_TCIA02_290_1',
            'HGG/Brats18_CBICA_ASW_1/Brats18_CBICA_ASW_1',
            'HGG/Brats18_TCIA01_460_1/Brats18_TCIA01_460_1',
            'HGG/Brats18_TCIA02_606_1/Brats18_TCIA02_606_1',
            'HGG/Brats18_TCIA01_201_1/Brats18_TCIA01_201_1',
            'HGG/Brats18_CBICA_ARW_1/Brats18_CBICA_ARW_1',
            'HGG/Brats18_TCIA01_147_1/Brats18_TCIA01_147_1',
            'HGG/Brats18_TCIA08_105_1/Brats18_TCIA08_105_1',
            'HGG/Brats18_CBICA_AMH_1/Brats18_CBICA_AMH_1',
            'HGG/Brats18_CBICA_ABM_1/Brats18_CBICA_ABM_1',
            'HGG/Brats18_CBICA_ABY_1/Brats18_CBICA_ABY_1',
            'HGG/Brats18_TCIA03_257_1/Brats18_TCIA03_257_1',
            'HGG/Brats18_TCIA02_331_1/Brats18_TCIA02_331_1',
            'HGG/Brats18_TCIA01_335_1/Brats18_TCIA01_335_1',
            'HGG/Brats18_TCIA01_401_1/Brats18_TCIA01_401_1',
            'HGG/Brats18_TCIA05_478_1/Brats18_TCIA05_478_1',
            'HGG/Brats18_CBICA_AAP_1/Brats18_CBICA_AAP_1',
            'HGG/Brats18_CBICA_BFB_1/Brats18_CBICA_BFB_1',
            'HGG/Brats18_CBICA_AXN_1/Brats18_CBICA_AXN_1',
            'HGG/Brats18_CBICA_AQQ_1/Brats18_CBICA_AQQ_1',
            'HGG/Brats18_CBICA_AXM_1/Brats18_CBICA_AXM_1',
            'HGG/Brats18_TCIA01_390_1/Brats18_TCIA01_390_1',
            'HGG/Brats18_TCIA04_192_1/Brats18_TCIA04_192_1',
            'HGG/Brats18_CBICA_ABB_1/Brats18_CBICA_ABB_1',
            'HGG/Brats18_CBICA_ASE_1/Brats18_CBICA_ASE_1',
            'HGG/Brats18_TCIA08_406_1/Brats18_TCIA08_406_1',
            'HGG/Brats18_CBICA_ABE_1/Brats18_CBICA_ABE_1',
            'HGG/Brats18_CBICA_AZD_1/Brats18_CBICA_AZD_1',
            'HGG/Brats18_TCIA04_111_1/Brats18_TCIA04_111_1',
            'HGG/Brats18_TCIA03_199_1/Brats18_TCIA03_199_1',
            'HGG/Brats18_TCIA02_222_1/Brats18_TCIA02_222_1',
            'HGG/Brats18_TCIA01_499_1/Brats18_TCIA01_499_1',
            'HGG/Brats18_2013_19_1/Brats18_2013_19_1',
            'HGG/Brats18_2013_22_1/Brats18_2013_22_1',
            'HGG/Brats18_TCIA08_218_1/Brats18_TCIA08_218_1',
            'HGG/Brats18_TCIA03_133_1/Brats18_TCIA03_133_1',
            'HGG/Brats18_2013_3_1/Brats18_2013_3_1',
            'HGG/Brats18_TCIA01_231_1/Brats18_TCIA01_231_1',
            'HGG/Brats18_TCIA06_165_1/Brats18_TCIA06_165_1',
            'LGG/Brats18_TCIA13_623_1/Brats18_TCIA13_623_1',
            'LGG/Brats18_TCIA13_624_1/Brats18_TCIA13_624_1',
            'LGG/Brats18_TCIA10_346_1/Brats18_TCIA10_346_1',
            'LGG/Brats18_TCIA13_654_1/Brats18_TCIA13_654_1',
            'LGG/Brats18_TCIA12_249_1/Brats18_TCIA12_249_1',
            'LGG/Brats18_TCIA10_637_1/Brats18_TCIA10_637_1',
            'LGG/Brats18_TCIA10_644_1/Brats18_TCIA10_644_1',
            'LGG/Brats18_TCIA10_299_1/Brats18_TCIA10_299_1',
            'LGG/Brats18_2013_28_1/Brats18_2013_28_1',
            'LGG/Brats18_TCIA10_202_1/Brats18_TCIA10_202_1',
            'LGG/Brats18_TCIA09_462_1/Brats18_TCIA09_462_1',
            'LGG/Brats18_TCIA10_266_1/Brats18_TCIA10_266_1',
            'LGG/Brats18_TCIA09_255_1/Brats18_TCIA09_255_1',
            'LGG/Brats18_TCIA13_633_1/Brats18_TCIA13_633_1',
            'LGG/Brats18_TCIA10_307_1/Brats18_TCIA10_307_1',
            'LGG/Brats18_2013_16_1/Brats18_2013_16_1',
            'LGG/Brats18_2013_6_1/Brats18_2013_6_1',
            'LGG/Brats18_TCIA12_298_1/Brats18_TCIA12_298_1',
            'LGG/Brats18_TCIA10_282_1/Brats18_TCIA10_282_1',
            'LGG/Brats18_TCIA13_642_1/Brats18_TCIA13_642_1',
            'LGG/Brats18_TCIA10_310_1/Brats18_TCIA10_310_1',
            'LGG/Brats18_TCIA10_442_1/Brats18_TCIA10_442_1',
            'LGG/Brats18_TCIA10_628_1/Brats18_TCIA10_628_1',
            'LGG/Brats18_TCIA09_312_1/Brats18_TCIA09_312_1',
            'LGG/Brats18_2013_24_1/Brats18_2013_24_1',
            'LGG/Brats18_TCIA10_261_1/Brats18_TCIA10_261_1',
            'LGG/Brats18_2013_29_1/Brats18_2013_29_1',
            'LGG/Brats18_TCIA10_351_1/Brats18_TCIA10_351_1',
            'LGG/Brats18_TCIA09_620_1/Brats18_TCIA09_620_1',
            'LGG/Brats18_TCIA10_413_1/Brats18_TCIA10_413_1',
            'LGG/Brats18_TCIA10_103_1/Brats18_TCIA10_103_1',
            'LGG/Brats18_TCIA10_639_1/Brats18_TCIA10_639_1',
            'LGG/Brats18_TCIA13_630_1/Brats18_TCIA13_630_1',
            'LGG/Brats18_2013_0_1/Brats18_2013_0_1',
            'LGG/Brats18_TCIA13_615_1/Brats18_TCIA13_615_1',
            'LGG/Brats18_TCIA10_130_1/Brats18_TCIA10_130_1',
            'LGG/Brats18_TCIA10_276_1/Brats18_TCIA10_276_1',
            'LGG/Brats18_TCIA13_650_1/Brats18_TCIA13_650_1',
            'LGG/Brats18_TCIA09_141_1/Brats18_TCIA09_141_1',
            'LGG/Brats18_TCIA13_653_1/Brats18_TCIA13_653_1',
            'LGG/Brats18_TCIA10_410_1/Brats18_TCIA10_410_1',
            'LGG/Brats18_2013_9_1/Brats18_2013_9_1',
            'LGG/Brats18_TCIA09_402_1/Brats18_TCIA09_402_1',
            'LGG/Brats18_TCIA10_408_1/Brats18_TCIA10_408_1']


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
           
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        if self.test_flag:
            image=out
            image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, image, path)
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label, path)

    def __len__(self):
        return len(self.database)

class BRATSDataset3D(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                print(root)
                files.sort()
                datapoint = dict()
                # extract all files as channels
                id = os.path.join(root.split('/')[3], (root.split('/'))[4], (root.split('/'))[4])
                for f in files:
                    
                    print(id)
                    print(f)
                    if ('mask' in f):
                        continue
                    print(f)
                    seqtype = f.split('_')[4].split('.')[0]
                    datapoint[seqtype] = os.path.join(root, f)
                
                if id in train_ids:
                    assert set(datapoint.keys()) == self.seqtypes_set, \
                        f'datapoint is incomplete, keys are {datapoint.keys()}'
                    self.database.append(datapoint)
        
    def __len__(self):
        print(len(self.database))
        return len(self.database) * 155

    def __getitem__(self, x):
        out = []
        n = x // 155
        slice = x % 155
        filedict = self.database[n]
        for seqtype in self.seqtypes:
            # print(filedict[seqtype])
            # print(7/0)
            sitk_img = sitk.ReadImage(filedict[seqtype])
            #nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            o = torch.tensor(sitk.GetArrayFromImage(sitk_img)[slice,:,:])
            #o = torch.tensor(nib_img.get_fdata())[:,:,slice]
            # if seqtype != 'seg':
            #     o = o / o.max()
            out.append(o)
        out = torch.stack(out)
        if self.test_flag:
            image=out
            # image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, image, path.split('.nii')[0] + "_slice" + str(slice)+ ".nii") # virtual path
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            # image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            # label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label, path.split('.nii')[0] + "_slice" + str(slice)+ ".nii") # virtual path



