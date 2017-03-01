require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts.lua') --read input args
opt = opts.parse(arg)

nClasses = opt.nClasses

utils = paths.dofile('util.lua') -- init data parallel functions
paths.dofile('model.lua') -- create model
opt.imageSize = model.imageSize or opt.imageSize
opt.imageCrop = model.imageCrop or opt.imageCrop

print(opt)

cutorch.setDevice(opt.GPU) --set default device
torch.manualSeed(opt.manualSeed)

paths.dofile(paths.concat(opt.data, 'getData.lua')) --Load data
paths.dofile('data_loader.lua') -- Parallel data loading functions
paths.dofile('train.lua')
paths.dofile('test.lua')

epoch = opt.epochNumber
bestAccuracy = 0.0

for i=1,opt.nEpochs do
   train()
   test()
   --If last run gave the best accuracy then save the model
   if bestAccuracy < testAcc then
      print('==>This epoch got the best accuracy, so saving the model')
      bestAccuracy = testAcc
      -- clear the intermediate states in the model before saving to disk
      -- this saves lots of disk space
      model:clearState()
      utils.saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
      torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
   end
   epoch = epoch + 1
end
