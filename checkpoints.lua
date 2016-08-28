--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local checkpoint = {}
local opt = require 'local_config'


function checkpoint.makeDataParallel(model, nGPU)
   if nGPU > 1 then
      --print('converting module to nn.DataParallelTable')
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local model_single = model
      model = nn.DataParallelTable(1)
      for i=1, nGPU do
         cutorch.setDevice(i)
         model:add(model_single:clone():cuda(), i)
      end
   end
   cutorch.setDevice(opt.GPU)

   return model
end

local function cleanDPT(module)
   -- This assumes this DPT was created by the function above: all the
   -- module.modules are clones of the same network on different GPUs
   -- hence we only need to keep one when saving the model to the disk.
   local newDPT = nn.DataParallelTable(1)
   cutorch.setDevice(opt.GPU)
   newDPT:add(module:get(1), opt.GPU)
   return newDPT
end


function checkpoint.loadDataParallel(filename, nGPU)
   if opt.backend == 'cudnn' then
      require 'cudnn'
   end
   local model = torch.load(filename)
   if torch.type(model) == 'nn.DataParallelTable' then
      return makeDataParallel(model:get(1):float(), opt.nGPU)
   else
         return checkpoint.makeDataParallel(model, opt.nGPU)
      -- error('The loaded model is not a Sequential or DataParallelTable module.')
   end
end

function checkpoint.latest(opt)
   if opt.resume == 'none' then
      return nil
   end

   -- local latestPath = paths.concat(opt.resume, 'latest.t7')
   local latestPath = 'latest2.t7'

   if not paths.filep(latestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(latestPath) -- Wrong
   local optimState = torch.load(paths.concat(latest.optimFile))
   return latest, optimState
end

function checkpoint.save(epoch, model, optimState, opt)
   -- Don't save the DataParallelTable for easier loading on other machines
   local bestModel = opt.bestModel
   local prefix = opt.prefix
   if torch.type(model) == 'nn.DataParallelTable' then
      model = checkpoint.cleanDPT(model)
   end
   model:clearState()
   local modelFile = paths.concat(prefix, 'model_' .. epoch .. '.t7')
   local optimFile = paths.concat(prefix, 'optimState_' .. epoch .. '.t7')

   torch.save(modelFile, model)
   torch.save(optimFile, optimState)
   torch.save('latest2.t7', {
      epoch = epoch,
      modelFile = modelFile,
      optimFile = optimFile,
   })

   if bestModel then
      torch.save('model_best.t7', model)
   end
end

return checkpoint