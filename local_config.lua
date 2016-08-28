--VGG_WEIGHTS_FOLDER = 'data/weights/depthnormals_nyud_vgg/'
--SEMLABELS_WEIGHTS_FOLDER = 'data/weights/semlabels_nyud40_vgg/'


-- Parameters
--FOLDER_PREFIX = VGG_WEIGHTS_FOLDER
require 'cutorch'
batchMode = true -- Keep it true because SpatialUnitNormalization not implemented
depthConcat = (batchMode) and 2 or 1
batchSize = 1
--local opt = require 'local_config'

deviceId = 4

cutorch.setDevice(deviceId)

-- print(depthConcat)

torch.manualSeed(2016)
cutorch.manualSeed(2016)
cutorch.manualSeedAll(2016)


local local_config = {}

-- Image Size for the network
local_config['iH'] = 224
local_config['iW'] = 224 

local_config['oH'] = 224
local_config['oW'] = 224

--local_config['Scale2_iH'] = 109
--local_config['Scale2_iW'] = 147


-- Default GPU
local_config['GPU'] = deviceId
local_config['nGPU'] = 1
local_config['save'] = '.'

local_config['LR'] = 0.0001
local_config['learningRateDecay'] = 0.00001
local_config['weightDecay'] = 0.0001
local_config['momentum'] = 0.9

local_config['batchSize'] = 128

-- local_config['nThreads'] = 2

return local_config
