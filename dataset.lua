require 'image'
require 'torch'
require 'cutorch'
-- require 'utils'
local conf=require 'local_config'
cutorch.setDevice(conf.GPU)
local pl = require('pl.import_into')()
--threads = require 'threads'
--conf = require 'local_config'
torch.setdefaulttensortype('torch.FloatTensor')
local dataset = torch.class("DataLoader")
local MIN_ID = 1
local MAX_ID = 180000
local TOTAL = MAX_ID - MIN_ID + 1
local train_image=torch.load("train_images.t7")
local test_image=torch.load("val_images.t7")
local test_label=torch.load("val_one_hot.t7")
local train_label=torch.load('train_one_hot.t7')
-- TRAIN_PATH = "/media/navneet/74E8952AE894EC1C/INKERS/Interns/himanshu/task1/try_googlenet/files/train_files/"

-- TEST_PATH="/media/navneet/74E8952AE894EC1C/INKERS/Interns/himanshu/task1/try_googlenet/files/test_files/"
-- VAL_PATH="/media/navneet/74E8952AE894EC1C/INKERS/Interns/himanshu/task1/try_googlenet/files/val_files/"
-- TRAIN_IMAGE_PATH="/media/navneet/74E8952AE894EC1C/INKERS/Interns/data/train_images/"
-- TEST_IMAGE_PATH="/media/navneet/74E8952AE894EC1C/INKERS/Interns/data/test_images/"
-- VAL_IMAGE_PATH="/media/navneet/74E8952AE894EC1C/INKERS/Interns/data/val_images/"



local function insertToTables(input, output, a, b)
	table.insert(input, a)
	table.insert(output,b)
	--print("inserted into tables")
end

function dataset:__init()
	self.INPUT_IMAGE = {}
	self.OUTPUT_ALL={}
end

function dataset:load_all_data()
		return train_image,train_label  
end
collectgarbage()
collectgarbage()

function dataset:createTrainTestSplit(ratio)
	local order =  torch.randperm(TOTAL):long()
	self._train_image = order:narrow(1,1,math.floor(ratio*TOTAL))
end
collectgarbage()
-- -- sampler, samples from the training set.
function dataset:sampleTrain(batchSize)
	local input = {}
	local outputs = {}
	order = torch.randperm(train_image:size(1)-1):long()
	num_batches = math.floor((train_image:size(1)-1)/batchSize) 
	local reorder_input_img = train_image:index(1,order:long())
	local reorder_output_map=train_label:index(1,order:long())
		
	for i=1,num_batches do 		
		local temp_in = reorder_input_img:narrow(1, batchSize*(i-1)+1, batchSize)
		local temp_out = reorder_output_map:narrow(1, batchSize*(i-1)+1, batchSize)
		insertToTables(input,outputs,temp_in,temp_out)
	end
	collectgarbage()
	return input, outputs
end
collectgarbage()

-- -- TODO Should be done once
function dataset:sampleTest(batchSize)
	local test_input = {}
	local test_outputs = {}
	local order = torch.randperm(test_image:size(1)-1):long()
	local num_batches = math.floor(test_image:size(1)/batchSize) -- TODO Correct for remaning batches
	local reorder_test_img = test_image:index(1, order:long())
	local reorder_test_map = test_label:index(1, order:long())
	
	for i=1,num_batches do 
		local test_temp_in = reorder_test_img:narrow(1, batchSize*(i-1)+1, batchSize)
		local test_temp_out = reorder_test_map:narrow(1, batchSize*(i-1)+1, batchSize)
		insertToTables(test_input,test_outputs,test_temp_in,test_temp_out)
		-- Depth Concat
	end
	collectgarbage()
	collectgarbage()
	return test_input, test_outputs
end
collectgarbage()

local load_image = function(path, size)
  local size  = size or 224
  local img   = image.load(path)
  local c,w,h = img:size(1), img:size(3), img:size(2)
  assert(c == 3)
  local min   = math.min(w, h)
  img         = image.crop(img, 'c', min, min)
  img         = image.scale(img, size)
  -- normalize (see inception.ipynb -> `ClassifyImageWithInception`)
  img:mul(255):clamp(0, 255):add(-117)
  return img:view(1, img:size(1), img:size(2), img:size(3))
end

train_name={
   "bedroom",
   "church",
   "classroom",
   "conference_room",
   "dining_room",
   "kitchen",
   "living_room",
   "restaurant",
   "tower"
}
function dataset:save()



			 local synsets = pl.utils.readlines('synsets.txt')
			 model:add(nn.SoftMax())
			 model:cuda()
			timer=torch.Timer()
			 load_time=0
			 predicting_time=0
			for i=2,20 do
					timer1 = torch.Timer() 				
					img="test"..i..'.jpg'

					f=load_image(img)
					--print('Time elapsed for loading iamge is ' .. timer1:time().real .. ' seconds')
					--print(img)
					load_time=load_time+timer1:time().real
					timer2 = torch.Timer()
					z=f:cuda()
					-- f,t=cutorch.getMemoryUsage(conf.GPU)
	 		
					local scores = model:forward(z)
      				scores = scores:float():squeeze()
					local _,ind = torch.sort(scores, true)	
					--print(i..train_name[ind[1]])			
					predicting_time=predicting_time+timer2:time().real
					--print('Time elapsed for predicting image is ' .. timer2:time().real .. ' seconds')
					-- f,t=cutorch.getMemoryUsage(conf.GPU)
	 		
					--print(ind)
					 -- print('\nRESULTS (top-9):')
					 -- print('----------------')
					 -- for i=1,9 do
					 --  local synidx = ind[i] -- synsets is 1-based
					 --   print(string.format("score = %f: %s (%d)", scores[ind[i]], train_name[synidx],ind[i]))
					 -- end
	 		
	 		-- m=((t-f)/(1024*1024))
	 		-- memory=m
	 		-- print("after forawrd pas",memory)
	 	end
	 	load_time=load_time/19
	 	predicting_time=predicting_time/19
	 	print("average loading time is "..load_time)
	 	print("average predicting time is "..predicting_time)
	 	print('Time elapsed for whole batch  is ' .. timer:time().real .. ' seconds')
	 		
	 		-- print(cutorch.getMemoryUsage(conf.GPU)[1]-cutorch.getMemoryUsage(conf.GPU)[2])
end        

 collectgarbage()
 collectgarbage()
 
 
