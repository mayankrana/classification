require 'paths'

local zipfile_name = 'cifar10torchsmall.zip'
local zipfile_path = paths.concat(opt.data, zipfile_name)

if (not paths.filep(zipfile_path)) then
   os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip -O ' .. zipfile_path)
   os.execute('unzip ' .. zipfile_path .. ' -d ' .. opt.data )
end
trainset = torch.load(paths.concat(opt.data, 'cifar10-train.t7'))
trainset.data = trainset.data:float()
trainset.data = trainset.data[{{1,1000},{},{},{}}]

testset  = torch.load(paths.concat(opt.data,'cifar10-test.t7'))
testset.data = testset.data:float()
testset.data = testset.data[{{1,1000},{},{},{}}]
classes  = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
