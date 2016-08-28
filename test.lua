require 'cutorch'
local opt = require 'local_config'

local inputs = torch.CudaTensor()
local labels = torch.CudaTensor(opt.batchSize,1)
collectgarbage()

function testBatch(inputsCPU, labelsCPU)
   -- batchNumber = batchNumber + opt.batchSize
   model:evaluate()
   inputs:resize(inputsCPU:size()):copy(inputsCPU:cuda())
   for i=1,128 do
      for j=1,9 do
        if labelsCPU[i][j]==1 then 
          
          labels[i]=j
        
        end
       end
    end  
   --labels:resize(labelsCPU:size()):copy(labelsCPU)

   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, labels)
   cutorch.synchronize()
   -- local pred = outputs:float()
   if labelsCPU ~= nil then
      return err, outputs
   else
      return outputs
   end
   collectgarbage()
 
   --eturn err, outputs
end
collectgarbage()
