require 'models'
require 'dataset'
require 'train'
require 'test'
require 'cutorch'
require 'threads'
require 'gnuplot'

local pl = require('pl.import_into')()

local opt = require 'local_config'
cutorch.setDevice(opt.GPU)
local checkpoint = require 'checkpoints'

function init_model()
	print("Loading Model...")
	timer = torch.Timer()
	model , optimParams=checkpoint.latest({resume='train2/'})
	
	if (model == nil) then
		timer = torch.Timer() -- the Timer starts to count now

		local m = models.googlenet()
		
		
		model = m:getModel():cuda()
		checkpoint.makeDataParallel(m:getModel(), opt.nGPU) -- Global
		print('Time elapsed for loading model is ' .. timer:time().real .. ' seconds')
		--print(model)
		optimParams = m:getOptimParameters() -- Global
		epoch = 0
		optimState = {
		    learningRate = opt.LR,
		    learningRateDecay = opt.learningRateDecay,
		    momentum = opt.momentum,
		    dampening = 0.0,
		    weightDecay = opt.weightDecay
		}
		print("New Model Successfully Created with default parameters")
		
	else 
		epoch = model.epoch
		print(model)
		model = checkpoint.loadDataParallel(model.modelFile)
		print('Time elapsed for loading model is ' .. timer:time().real .. ' seconds')
	end

	return model, optimState, epoch
end


function init_dataset(split)
	local dataLoader = DataLoader()
	--print('Loading Data...')
	local timer = torch.Timer()
	dataLoader:load_all_data()

	--print("Time take to load data ", timer:time().real)

	if split~=nil then
		-- Split dataset
	   dataLoader:createTrainTestSplit(split)
	end



	return dataLoader
end

function init_trainer()
	criterion =nn.CrossEntropyCriterion():cuda()
	print('Accumulating Parameters...')
	--print(model)
    parameters, gradParameters = model:parameters()
--    print("parameters ka type hai ",parameters)             

   -- print("parameters in main is ",#parameters)
   -- print("grad parameters in main is ",#gradParameters)
end

function init_training()
	d = init_dataset(1)

	print('Training ...')
	trainingParams = {
		saveModel = -100, -- -1 For No Automatic Saving
		maxEpochs =0,
		tester = testBatch,

		checkpoint = checkpoint,
		saveBest = true,
		saveBestFrom = 100, -- TODO
		valid = true
	}

	train(d, trainingParams)

end

function init_testing()
	 d = init_dataset()
	-- criterion = nn.CrossEntropyCriterion():cuda() -- TODO Set 

	-- print('Testing and saving files')
	-- local dataIn, dataOut = d:sampleTest(opt.batchSize)
	
	-- 		err, predOut = testBatch(dataIn[1], dataOut[1])
	-- 		 print("Error on batch ", err)

	-- 		d:save(dataIn[1], predOut:float(), "./", dataOut[1])
	d:save()
	
	
end


function main()
	model, optimState, epoch = init_model()

	 --init_trainer()
	 --init_training()

	init_testing()

end

main()
















