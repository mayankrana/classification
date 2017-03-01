require 'nn'
require 'cunn'
require 'optim'

-- 1. Create Network
-- 1.1 If preloading option is set, preload weights from existing models appropriately
if opt.retrain ~= 'none' then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   model = utils.loadDataParallel(opt.retrain, opt.nGPU) -- defined in util.lua
else
   paths.dofile('models/' .. opt.netType .. '.lua')
   print('=> Creating model from file: models/' .. opt.netType .. '.lua')
   model = createModel(opt.nGPU) -- for the model creation code, check the models/ folder
   --Initialize weights
   if opt.wInit then
      for indx,module in pairs(model:findModules('nn.SpatialConvolution')) do
         module.weight:normal(0,math.sqrt(2/(module.kW*module.kH*module.nOutputPlane)))
      end
   end
   require 'cudnn'
   cudnn.convert(model, cudnn)
end

-- 2. Create Criterion
criterion = nn.ClassNLLCriterion()

print('=> Model')
print(model)

print('=> Criterion')
print(criterion)

-- 3. Convert model to CUDA
--print('==> Converting model to CUDA')
-- model is converted to CUDA in the init script itself
-- model = model:cuda()
criterion:cuda()

collectgarbage()
