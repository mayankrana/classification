function createModel(nGPU)

   local cfg = {64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   --out_width  = floor((in_width  + 2*padding - kW) / dW + 1)

   local features = nn.Sequential()
   local inChannels = 3
   for k,v in ipairs(cfg) do
      if v == 'M' then
         features:add(nn.SpatialMaxPooling(2,2,2,2))
      else
         features:add(nn.SpatialConvolution(inChannels,v, 3,3, 1,1, 1,1))
         features:add(nn.ReLU(true))
         inChannels = v
      end
   end

   features:cuda()
   features = utils.makeDataParallel(features, opt.nGPU)

   local classifier = nn.Sequential()
   classifier:add(nn.View(512))
   classifier:add(nn.Linear(512, 512))
   classifier:add(nn.Threshold(0,1e-6))
   classifier:add(nn.BatchNormalization(512, 1e-3))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(512, nClasses))
   classifier:add(nn.Threshold(0,1e-6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.LogSoftMax())
   classifier:cuda()

   local model = nn.Sequential()
   model:add(features):add(classifier)
   model.inputSize = 32
   model.sampleSize = 32

   return model
end
