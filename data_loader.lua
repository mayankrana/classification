local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local options = opt
local train_set = trainset
local test_set = testset
local t_classes = classes

if opt.nDataThreads > 0 then
   dataLoaders = Threads(opt.nDataThreads,
                         function() require 'torch';
                            local Threads = require 'threads'
                            Threads.serialization('threads.sharedserialize')
                         end,
                         function(idx)
                            opt = options; tid = idx;
                            local seed = opt.manualSeed + idx; torch.manualSeed(seed);
                            trainset = train_set
                            testset = test_set
                            classes = t_classes
                            print(string.format('Starting workers with id:%d, seed:%d', tid, seed))
                            paths.dofile('dataprep.lua')
                         end
   );
else
   paths.dofile('dataprep.lua')
   dataLoaders = {}
   function dataLoaders:addjob(f1, f2) f2(f1()) end
   function dataLoaders:synchronize() end
end

nClasses = #classes
nTest = testset.data:size(1)

-- nTest=nil
-- nClasses=10
-- dataLoaders:addjob(function() return testset.data:size(1) end, function(v) nTest = v end)
-- dataLoaders:synchronize()
-- assert(nTest)
