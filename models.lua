require 'cunn'
require 'cutorch'
require 'local_config'
require 'nnx';
require 'image';
require 'dpnn'
require 'cudnn'
require 'models/model'
--require 'models/cnn'
require 'models/googlenet'

--require 'utils'
--require 'lib'
--require 'shortcuts'
--require 'models/vgg_scale1'
--require 'models/resnet_scale1'
--require 'models/scale2'
--require 'models/scale3'
--require 'models/vgg_depthnormals'
--require 'models/resnet_depthnormals'
--require 'models/sl_vgg_scale1'
--require 'models/sl_scale2'
--require 'models/sl_scale3'
--require 'models/vgg_labels'


require 'checkpoints'
-- TODO Use torch.include instead
-- print(models)
-- model = models.VGG_DepthNormals():getModel()

-- Loading Saving 13 Secs in total
-- print("Saving Model. Please wait")
-- model:clearState()
-- torch.save("filename.t7", model)

-- Loading
-- model = torch.load("data/filename.t7")
-- print(model)


return nil
