require 'optim'
require 'cutorch'
require 'paths'
require 'os'
--require 'data_augmentation'
local opt = require 'local_config'
require 'checkpoints'
require 'image'


-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, string.format('train_%d.log', os.time())))
local batchNumber
local top1_epoch, loss_epoch

function train(d, params)
   print('==> doing epoch on training data:')
   print("==> Batch Epoch # " .. epoch)


   batchNumber = 0
   cutorch.synchronize()
   local checkpoint = params.checkpoint

   tester = params.tester

   -- set the dropouts to training mode
   model:training()

   local tm = torch.Timer()
   top1_epoch = 0
   loss_epoch = 0

   bestError = 10000000000
         losses = {}
  for i=1,params.maxEpochs do
      -- queue jobs to data-workers

      --local LinearBatchSize = math.floor(2+i/params.maxEpochs*(opt.batchSize-1))
      local LinearBatchSize = opt.batchSize
      print("Batch Size", LinearBatchSize)
      dataIn, dataOut = d:sampleTrain(LinearBatchSize)
      --print("in",#dataIn[1])
       --print("out",#dataOut[1])
     -- assert(#dataIn==#dataOut)

       opt.epochSize = #dataIn


      for j=1, #dataIn do
        --print("calling train batch")
        --print(#dataIn[j],#dataOut[j])
        trainBatch(dataIn[j], dataOut[j])
  
      
  
      end

      epoch = epoch + 1

      if epoch%params.saveModel == 0 and params.saveModel~=-1 then
        model:clearState()
        print("Saving Model")
       checkpoint.save(epoch, model, optimParams, {prefix = "train2/"})
      end

      batchNumber = 0

       -- donkeys:synchronize()
       cutorch.synchronize()

       loss_epoch = loss_epoch / opt.epochSize


       if params.valid then
         validIn, validOut = d:sampleTest(opt.batchSize)---TODO sampletest
         validError = 0
         -- Validation 
         for j = 1,#validIn do
          validError = validError + tester(validIn[j], validOut[j])
         end

         validError = validError / #validIn
         print("validError,bestError",validError,bestError)

         if validError<bestError and params.saveBest then
          model:clearState()
          --print("hi")
          print("Saving best Model")

          bestError = validError
         else
          validError = -1
         end
  end
  collectgarbage()
  


       trainLogger:add{
          ['avg loss (train set)'] = loss_epoch,
          ['avg loss (valid set)'] = validError
       }
       print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                              .. 'average loss (per batch): %.2f \t '
                              .. 'average loss (valid batch): %.2f \t ',
                           epoch, tm:time().real, loss_epoch, validError))
print("\n")

       --loss_epoch = 0
       -- save model
       collectgarbage()

       -- clear the intermediate states in the model before saving to disk
       -- this saves lots of disk space
       model:clearState()
       -- TODO Use new saving API here.
       -- saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
       -- torch.save(paths.concat(opt.save, 'optimParams_' .. epoch .. '.t7'), optimParams)
--int("end of train para")

end
-- print("losses in train are ",losses)
 end-- of train()
-------------------------------------------------------------------------------------------







-- GPU inputs (preallocate)
local inputs = torch.CudaTensor(opt.batchSize,3,224,224)
local labels = torch.CudaTensor(opt.batchSize,1):fill(0)

local timer = torch.Timer()
local dataTimer = torch.Timer()

function trainBatch(inputsCPU, labelsCPU)
  ---print("insinde train batcg")
    model:training()
    -- cutorch.synchronize()
    collectgarbage()
    local dataLoadingTime = dataTimer:time().real
    --print("1",#inputsCPU[1])
    --print("2",#inputsCPU[2])

    timer:reset()
    local b=inputsCPU[1]:size()
   --print(#inputsCPU[1])
   --print(#inputs[1])

    -- transfer over to GPU
    -- print("labels ka size hai ", #labels)

   inputs=inputsCPU:cuda()
   --print("lables of cpu are ",#labelsCPU)
   for i=1,128 do
      for j=1,9 do
        if labelsCPU[i][j]==1 then 
          
          labels[i]=j
        
        end
       end
    end      
    -- print("labels are ",labels)
    --print("labels ",#labelsCPU)
     -- labels=labelsCPU:cuda()
          local err, outputs

    model:zeroGradParameters()
    --print(#inputs)
    local outputs=torch.CudaTensor()
    outputs = model:forward(inputs)
    -- print(#outputs)
     --print(#labels)
    --print("outptu size is ",outputs:size())
     err = criterion:forward(outputs,labels)
     table.insert(losses ,err)
    local gradOutputs = criterion:backward(outputs,labels)
    model:backward(inputs, gradOutputs)
    -- feval = function(x)
    --     return err, gradParameters
    -- end

    for i=1, #parameters do
      local feval2 = function(x)
        return err, gradParameters[i]
      end
      
      optim.sgd(feval2, parameters[i], optimParams[i])
    end
collectgarbage()
    -- optim.sgd(feval, parameters, optimParams)
    -- optim.adagrad(feval, parameters, optimParams)

    -- DataParallelTable's syncParameters
    if model.needsSync then
        model:syncParameters()
    end


    cutorch.synchronize()
    batchNumber = batchNumber + 1
    loss_epoch = loss_epoch + err

--print(opt.epochSize)
    -- Checking whether the memory doesnot changes while training
    -- assert(parameters:storage() == model:parameters()[1]:storage())

  --local clr = optimParams.learningRate / (1 + epoch*optimParams.learningRateDecay)
    local clr = -1
    print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, err,
        clr, dataLoadingTime))
    dataTimer:reset()
    collectgarbage()
end
--end

-- trainBatch(torch.rand(4,3,100,100), torch.rand(4,3,100,100))

