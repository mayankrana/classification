require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

dataAugmentationOpt = {
   lcn = true,
   hflip = true,
   vflip = false,
   rotate = false,
   random_crop = true --always true for now
}

local loadSize = {3, opt.imageSize, opt.imageSize}
local sampleSize = {3, opt.cropSize, opt.cropSize}
local mean = {}
local std = {}

--TODO Write code to load images if trainset and testset doesn't exist
local function loadImage(path)
   local input = image.load(path,3,'float')
   --Resize bigger dimension to loadSize keeping the aspect ratio
   if input:size(3) < input:size(2) then
      input = image.scale(input, loadSize[2], loadSize[3] * input:size(2) / input:size(3))
   else
      input = image.scale(input, loadSize[2] * input:size(3) / input:size(2), loadSize[3])
   end
   return input
end

function processTrain(input)

   collectgarbage()
   local out = input

   if dataAugmentationOpt.random_crop then
      local h1 = math.ceil(torch.uniform(0, input:size(2)-sampleSize[2]))
      local w1 = math.ceil(torch.uniform(0, input:size(3)-sampleSize[3]))
      out = image.crop(input, w1, h1, w1+sampleSize[3], h1+sampleSize[2])
   end

   if dataAugmentationOpt.hflip then
      if torch.uniform() > 0.5 then out = image.hflip(out); end --horizontal flip
   end

   if dataAugmentationOpt.vflip then
      if torch.uniform() > 0.5 then out = image.vflip(out); end --vertical flip
   end

   if dataAugmentationOpt.lcn and mean and std then
      --mean, std = torch.load('meanstd.t7')
      for i=1,3 do
         out[{{i},{},{}}]:add(-mean[i])
         out[{{i},{},{}}]:div(std[i])
      end
   end

   return out
end

function processTest(input, inputSize)
   --Do a center patch crop
   local h1 = math.ceil((input:size(2)-sampleSize[2])/2)
   local w1 = math.ceil((input:size(3)-sampleSize[3])/2)
   local out = image.crop(input, w1, h1, w1+sampleSize[3], h1+sampleSize[2])

   if dataAugmentationOpt.lcn and mean and std then
      for i=1,3 do
         out[{{i},{},{}}]:add(-mean[i])
         out[{{i},{},{}}]:div(std[i])
      end
   end
   return out
end

--expects global training dataset
function getTrainingMiniBatch(batchSize)
   local data = torch.Tensor(batchSize, sampleSize[1], sampleSize[2], sampleSize[3])
   local labels = torch.Tensor(batchSize)
   for i=1, batchSize do
      local idx = torch.random(1, trainset.data:size(1))
      data[i] = processTrain(trainset.data[idx]:clone())
      labels[i] = trainset.label[idx]
   end
   return data, labels
end

function getTestMiniBatch(idxStart, idxEnd)
   local data = torch.Tensor(idxEnd-idxStart+1, sampleSize[1], sampleSize[2], sampleSize[3])
   local labels = torch.Tensor(idxEnd-idxStart+1)
   for i=idxStart, idxEnd do
      local im = processTest(testset.data[i]:clone())
      data[i-idxStart+1] = im
      labels[i-idxStart+1] = testset.label[i]
   end
   return data, labels
end

--TODO For very big input dataset, only compute meanstd from a sample set
--Expects trainset and testset are already loaded
function computeMeanStdDev()
   for i=1,3 do
      mean[i] = trainset.data[{{},{i},{},{}}]:mean()
      std[i] = trainset.data[{{},{i},{},{}}]:std()
   end
   local meanstd = {}
   meanstd.mean = mean
   meanstd.std  = std
   torch.save(paths.concat(opt.cache, 'meanstd.t7'), meanstd)
end

local meanstdPath = paths.concat(opt.cache,'meanstd.t7')
if not paths.filep(meanstdPath) then
    computeMeanStdDev()
else
  local meanstd = torch.load(meanstdPath)
  mean = meanstd.mean
  std  = meanstd.std
end
