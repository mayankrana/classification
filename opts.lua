local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------

    cmd:option('-cache', './cifar/checkpoint/', 'subdirectory in which to save/log experiments')
    cmd:option('-data', 'data/cifar/', 'Home of dataset')
    cmd:option('-manualSeed',         1, 'Manually set seed')
    cmd:option('-GPU',                1, 'Default preferred GPU ID')
    cmd:option('-nGPU',               2, 'Number of GPUs to use by default')
    ------------- Data options ------------------------
    cmd:option('-nDataThreads',       2, 'number of data loading threads to initialize')
    cmd:option('-imageSize',         32,    'Smallest side of the resized image')
    cmd:option('-cropSize',          32,    'Height and Width of image crop to be used as input layer')
    cmd:option('-nClasses',        10, 'number of classes in the dataset')
    ------------- Training options --------------------
    cmd:option('-nEpochs',         55,    'Number of total epochs to run')
    cmd:option('-epochSize',       100, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       128,   'mini-batch size (1 = pure stochastic)')
    ---------- Optimization options ----------------------
    cmd:option('-LR',              0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     5e-4, 'weight decay')
    cmd:option('-decay',           0.5, 'weight decay')
    cmd:option('-learningRateStep', 10, 'apply decay after these number of iterations')
    ---------- Model options ----------------------------------
    cmd:option('-netType',     'vgg', 'Options: vgg')
    cmd:option('-retrain',     'none', 'provide path to model to retrain with')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
    cmd:option('-wInit',       true, 'init weights according to MSR paper')
    cmd:text()

    local opt = cmd:parse(arg or {})
    -- add commandline specified options
    -- opt.save = paths.concat(opt.cache,
    --                         cmd:string(opt.netType, opt,
    --                                    {netType=true, retrain=false, optimState=false, cache=tru    --e, data=true}))
    -- add date/time
    opt.save = paths.concat(opt.cache, opt.netType .. '/' .. os.date():gsub(' ',''))

    print('Saving everything to: ' .. opt.save)
    os.execute('mkdir -p ' .. opt.save)
    return opt
end

return M
